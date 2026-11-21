"""Physical domain randomization and privileged observations for RMA stage one."""

import torch

from isaaclab.actuators import DCMotor
from isaaclab.utils.math import euler_xyz_from_quat

def _foot_shape_indices(robot, foot_ids):
    """Translate articulation link IDs into the PhysX per-shape material layout."""
    if len(foot_ids) != 4:
        raise ValueError("SYSU219 must have four foot links.")
    index = 0
    shapes = []
    for body_id, path in enumerate(robot.root_physx_view.link_paths[0]):
        count = robot._physics_sim_view.create_rigid_body_view(path).max_shapes
        if body_id in foot_ids:
            if count == 0:
                raise ValueError(f"Missing collision shape on {path}")
            shapes.extend(range(index, index + count))
        index += count
    if index != robot.root_physx_view.max_shapes:
        raise RuntimeError("The link and articulation collision shape layouts differ.")
    return shapes


class PaperDCMotor(DCMotor):
    """Apply the sampled motor response before the actual SYSU219 PD/saturation model."""

    def compute(self, control_action, joint_pos, joint_vel):
        if hasattr(self, "strength"):
            # The authors' released Env.hpp scales the target offset from the
            # reference pose (pTarget12_.cwiseProduct(motor_strength)).
            target = control_action.joint_positions
            offset = target - self.reference_position
            control_action.joint_positions = self.reference_position + self.strength * offset
        return super().compute(control_action, joint_pos, joint_vel)


class _PaperDynamics:
    def __init__(self, env):
        self.robot = env.scene["robot"]
        self.foot_ids, _ = self.robot.find_bodies(".*_foot")
        self.foot_shapes = _foot_shape_indices(self.robot, self.foot_ids)
        (self.trunk,), _ = self.robot.find_bodies("trunk")
        self.view = self.robot.root_physx_view
        self.masses = self.view.get_masses().clone()
        self.coms = self.view.get_coms().clone()
        self.materials = self.view.get_material_properties().clone()
        self.nominal_mass = self.masses[:, self.trunk].clone()
        self.nominal_com = self.coms[:, self.trunk, :2].clone()
        self.actuator = self.robot.actuators["base_legs"]
        self.joints = self.actuator.joint_indices
        self.actuator.reference_position = self.robot.data.default_joint_pos[:, self.joints].clone()
        self.actuator.strength = torch.ones_like(self.actuator.reference_position)
        self.factors = torch.zeros(env.num_envs, 16, device=env.device)

    def sample(self, env, env_ids):
        if env_ids.numel() == 0:
            return
        ids = env_ids.to("cpu", dtype=torch.long)
        iteration = getattr(env, "paper_iteration", 0)
        start, end = env.cfg.disturbance_start_iteration, env.cfg.disturbance_full_iteration
        scale = min(1.0, max(0.0, (iteration - start) / (end - start)))
        # CPU PhysX tensor setters take CPU data/indices, even for GPU simulation.
        n = len(ids)
        u = torch.rand(n, 18, device="cpu")
        payload = u[:, 0] * (env.cfg.max_payload_kg * scale)
        # Table I reports COM in centimeters. The two labeled shifts are x and y.
        com = (2 * u[:, 1:3] - 1) * (0.0015 * scale)
        strength = 1 + (2 * u[:, 3:15] - 1) * (0.1 * scale)
        # The ground's multiply-combine material is (1,1), so these values are
        # the contact coefficients for each of the four feet.
        friction = 1 + scale * (0.05 + u[:, 15] * (4.5 - 0.05) - 1)
        kp = 55 + scale * (50 + 10 * u[:, 16] - 55)
        kd = 0.8 + scale * (0.4 + 0.4 * u[:, 17] - 0.8)

        self.masses[ids, self.trunk] = self.nominal_mass[ids] + payload
        self.coms[ids, self.trunk, :2] = self.nominal_com[ids] + com
        self.materials[ids[:, None], self.foot_shapes, :2] = friction[:, None, None]
        self.view.set_masses(self.masses, ids)
        self.view.set_coms(self.coms, ids)
        self.view.set_material_properties(self.materials, ids)
        # PhysX may round mass/position/material values. Read them back for the
        # teacher so labels reflect the simulated robot, including after resets.
        mass_applied = self.view.get_masses()[ids, self.trunk] - self.nominal_mass[ids]
        com_applied = self.view.get_coms()[ids, self.trunk, :2] - self.nominal_com[ids]
        mu_applied = self.view.get_material_properties()[ids[:, None], self.foot_shapes, :2]
        if not (
            torch.allclose(mass_applied, payload, atol=1e-5)
            and torch.allclose(com_applied, com, atol=1e-6)
            and torch.allclose(mu_applied, friction[:, None, None])
        ):
            raise RuntimeError("PhysX did not apply the requested RMA payload, COM or foot friction")

        self.actuator.strength[env_ids] = strength.to(env.device)
        self.actuator.stiffness[env_ids] = kp[:, None].to(env.device)
        self.actuator.damping[env_ids] = kd[:, None].to(env.device)
        applied = torch.cat((mass_applied[:, None], com_applied, strength, mu_applied[:, 0, :1]), dim=1)
        self.factors[env_ids] = applied.to(env.device)


def reset_episode(env, env_ids):
    """Pick a new terrain patch and draw the physical context for the episode."""
    if not hasattr(env, "_paper_dynamics"):
        env._paper_dynamics = _PaperDynamics(env)
    terrain = env.scene.terrain
    cfg = terrain.cfg.terrain_generator
    n = len(env_ids)
    origins = terrain.env_origins[env_ids].clone()
    # One global mesh supplies randomly sampled, fixed-difficulty terrain
    # profiles. Keep a half patch around the spawn for the 10-second walk.
    span = origins.new_tensor([(cfg.num_rows - 1) * cfg.size[0], (cfg.num_cols - 1) * cfg.size[1]])
    origins[:, :2] = (torch.rand(n, 2, device=env.device) - 0.5) * span
    robot = env.scene["robot"]
    # As in the authors' released reset, use the ground at the base XY,
    # adding this robot's configured standing height.
    ground = terrain.heights(origins)
    origins[:, 2] = ground
    terrain.env_origins[env_ids] = origins
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :3] += origins
    robot.write_root_pose_to_sim(state[:, :7], env_ids)
    robot.write_root_velocity_to_sim(state[:, 7:], env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids], robot.data.default_joint_vel[env_ids], env_ids=env_ids
    )
    env._paper_dynamics.sample(env, env_ids)


def resample_during_episode(env, env_ids):
    candidates = (torch.rand(env.num_envs, device=env.device) < 0.004) & (env.episode_length_buf > 0)
    env._paper_dynamics.sample(env, candidates.nonzero().flatten())


def privileged_factors(env):
    if not hasattr(env, "_paper_dynamics"):
        # The observation manager queries dimensions once before the first reset.
        return torch.zeros(env.num_envs, 17, device=env.device)
    dyn = env._paper_dynamics
    terrain = env.scene.terrain
    feet = dyn.robot.data.body_link_pos_w[:, dyn.foot_ids]
    local_heights = terrain.heights(feet)
    # A ray beyond the finite mesh has no hit. Keep the height observation finite.
    local_heights = torch.where(torch.isfinite(local_heights), local_heights, 0.0)
    height = (local_heights * 10).round().amax(dim=1, keepdim=True) / 10
    return torch.cat((dyn.factors, height), dim=1)


def fallen(env):
    robot = env.scene["robot"]
    root = robot.data.root_link_pos_w
    terrain_z = env.scene.terrain.heights(root)
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
    low = torch.isfinite(terrain_z) & ((root[:, 2] - terrain_z) < 0.28)
    return low | (roll.abs() > 0.4) | (pitch.abs() > 0.2)
