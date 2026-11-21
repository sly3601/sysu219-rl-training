"""SYSU219 realization of the state, action and step reward in RMA §III-A/IV-A."""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ManagerTermBase, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

from .paper_contact import PaperFootContactSensor


FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def configure_paper_foot_sensors(scene):
    """PhysX filtered contact reporting requires a separate sensor body per environment."""
    for foot in FOOT_NAMES:
        setattr(
            scene,
            f"ground_contact_{foot}",
            ContactSensorCfg(
                class_type=PaperFootContactSensor,
                prim_path=f"{{ENV_REGEX_NS}}/Robot/sysu219/{foot}",
                filter_prim_paths_expr=[f"{scene.terrain.prim_path}/terrain/mesh"],
                track_friction_forces=True,
            ),
        )


def _ground_reactions(env):
    """Vector sum of normal and tangential foot-ground forces in world axes, in N."""
    forces = []
    for foot in FOOT_NAMES:
        data = env.scene[f"ground_contact_{foot}"].data
        forces.append((data.force_matrix_w[:, 0] + data.friction_forces_w[:, 0]).sum(dim=1))
    return torch.stack(forces, dim=1)


def _foot_contacts(env):
    # Positive contact force is the binary contact criterion in the linked author implementation.
    return torch.stack([
        env.scene[f"ground_contact_{foot}"].data.force_matrix_w[:, 0].abs().sum(dim=(1, 2)) > 0
        for foot in FOOT_NAMES
    ], dim=1)


def _roll_pitch(env):
    roll, pitch, _ = euler_xyz_from_quat(env.scene["robot"].data.root_link_quat_w)
    return torch.stack((roll, pitch), dim=1)


def _foot_indicators(env):
    return _foot_contacts(env).float()


def _privileged(env):
    from .paper_dynamics import privileged_factors

    return privileged_factors(env)


@configclass
class PaperObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_positions = ObservationTermCfg(func=mdp.joint_pos)
        joint_velocities = ObservationTermCfg(func=mdp.joint_vel)
        roll_pitch = ObservationTermCfg(func=_roll_pitch)
        foot_contacts = ObservationTermCfg(func=_foot_indicators)
        previous_action = ObservationTermCfg(func=mdp.last_action)

    @configclass
    class PrivilegedCfg(ObservationGroupCfg):
        environment = ObservationTermCfg(func=_privileged)

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class PaperActionsCfg:
    # a_t = desired joint angles (radians). The 2021 paper specifies no action offset.
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, offset=0.0, use_default_offset=False,
    )


@configclass
class PaperCommandsCfg:
    """The paper rewards forward progress directly."""


def penalty_multiplier(iteration: int) -> float:
    """Closed form of k[0]=0.03, k[t+1]=k[t]**0.997 (supplement §II)."""
    return 0.03 ** (0.997 ** iteration)


def _penalty_rate(env):
    # RewardManager multiplies each term by step_dt. The paper defines one reward per control step.
    return penalty_multiplier(getattr(env, "paper_iteration", 0)) / env.step_dt


def _forward(env):
    return env.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(max=0.35) / env.step_dt


def _lateral_and_rotation(env):
    data = env.scene["robot"].data
    return (data.root_link_lin_vel_b[:, 1].square() + data.root_link_ang_vel_b[:, 2].square()) / env.step_dt


class _Work(ManagerTermBase):
    """Absolute dot product of applied torque and joint displacement over one control step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_q = env.scene["robot"].data.joint_pos.clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.previous_q[ids] = self._env.scene["robot"].data.joint_pos[ids]

    def __call__(self, env):
        data = env.scene["robot"].data
        work = (data.applied_torque * (data.joint_pos - self.previous_q)).sum(dim=1).abs()
        self.previous_q.copy_(data.joint_pos)
        return work * _penalty_rate(env)


class _DifferencePenalty(ManagerTermBase):
    """Squared difference from the previous control step, with per-episode history."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros_like(self.measure(env))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0.0

    def __call__(self, env):
        value = self.measure(env)
        cost = (value - self.previous).reshape(env.num_envs, -1).square().sum(dim=1)
        self.previous.copy_(value)
        return cost * _penalty_rate(env)


class _GroundImpact(_DifferencePenalty):
    @staticmethod
    def measure(env):
        # §III-A calls f a ground reaction force. The authors' later Env.hpp
        # uses a magnitude per foot. PhysX reports the net vector per foot here.
        return torch.linalg.vector_norm(_ground_reactions(env), dim=-1)


class _GroundImpulse(_DifferencePenalty):
    """Variation in each foot's net contact impulse (N s) at successive control steps.

    PhysX divides the last physics substep's normal and tangential impulses by
    physics_dt when filling the foot sensors. Tangential impulses are reported
    at friction anchors without a mapping to individual normal contacts. The
    net impulse magnitude equals the sum of contact-impulse magnitudes for a
    single contact; multiple differently directed impulses can cancel.
    """

    @staticmethod
    def measure(env):
        return torch.linalg.vector_norm(_ground_reactions(env), dim=-1) * env.physics_dt


class _TorqueSmoothness(_DifferencePenalty):
    @staticmethod
    def measure(env):
        return env.scene["robot"].data.applied_torque


def _action_magnitude(env, stance_relative: bool = False):
    action = env.action_manager.action
    if stance_relative:
        # Phase 1 commands all joints in articulation order, in absolute radians.
        # Recover the later authors' policy coordinates before motor-strength randomization.
        action = (action - env.scene["robot"].data.default_joint_pos) / 0.4
    return action.square().sum(dim=1) * _penalty_rate(env)


def _joint_speed(env):
    return env.scene["robot"].data.joint_vel.square().sum(dim=1) * _penalty_rate(env)


def _orientation(env):
    return _roll_pitch(env).square().sum(dim=1) * _penalty_rate(env)


def _vertical_speed(env):
    # §III-A item 9 is titled "Z Acceleration", but its printed expression is -|v_z|².
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square() * _penalty_rate(env)


def _foot_slip(env, asset_cfg):
    speeds_sq = env.scene["robot"].data.body_link_lin_vel_w[:, asset_cfg.body_ids].square().sum(dim=2)
    return (speeds_sq * _foot_contacts(env)).sum(dim=1) * _penalty_rate(env)


@configclass
class PaperRewardsCfg:
    forward = RewardTermCfg(func=_forward, weight=20.0)
    lateral_and_rotation = RewardTermCfg(func=_lateral_and_rotation, weight=-21.0)
    work = RewardTermCfg(func=_Work, weight=-0.002)
    ground_impact = RewardTermCfg(func=_GroundImpact, weight=-0.02)
    torque_smoothness = RewardTermCfg(func=_TorqueSmoothness, weight=-0.001)
    action_magnitude = RewardTermCfg(func=_action_magnitude, weight=-0.07)
    joint_speed = RewardTermCfg(func=_joint_speed, weight=-0.002)
    orientation = RewardTermCfg(func=_orientation, weight=-1.5)
    vertical_speed = RewardTermCfg(func=_vertical_speed, weight=-2.0)
    foot_slip = RewardTermCfg(
        func=_foot_slip, weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_NAMES), preserve_order=True)},
    )
