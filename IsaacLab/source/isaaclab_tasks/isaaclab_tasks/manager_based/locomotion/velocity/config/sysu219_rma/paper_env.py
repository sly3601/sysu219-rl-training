"""SYSU219 port of the published RSS 2021 RMA first-stage task."""

import math
from pathlib import Path

import torch

import carb
from isaacsim.core.utils.extensions import get_extension_path_from_name

import isaaclab.envs.mdp as mdp
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.sysu219 import SYSU219_CFG

from ...velocity_env_cfg import MySceneCfg
from . import paper_dynamics as dynamics
from .paper_mdp import (
    PaperActionsCfg,
    PaperCommandsCfg,
    PaperObservationsCfg,
    PaperRewardsCfg,
    _GroundImpulse,
    configure_paper_foot_sensors,
)
from .paper_terrain import PaperFractalTerrain, PaperReplayTerrainImporter, PaperTerrainImporter


@configclass
class PaperEventsCfg:
    reset = EventTermCfg(func=dynamics.reset_episode, mode="reset")
    # The event runs once per policy step. Its own independent Bernoulli draw
    # for each environment implements Table I's probability per step.
    dynamics_change = EventTermCfg(
        func=dynamics.resample_during_episode, mode="interval", interval_range_s=(0.01, 0.01), is_global_time=True,
    )


@configclass
class PaperTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fall = TerminationTermCfg(func=dynamics.fallen)


@configclass
class SYSU219RMAPaperEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    actions: PaperActionsCfg = PaperActionsCfg()
    commands: PaperCommandsCfg = PaperCommandsCfg()
    observations: PaperObservationsCfg = PaperObservationsCfg()
    rewards: PaperRewardsCfg = PaperRewardsCfg()
    events: PaperEventsCfg = PaperEventsCfg()
    terminations: PaperTerminationsCfg = PaperTerminationsCfg()
    curriculum = None
    # The physics substep and the endpoints of the linear perturbation
    # curriculum are explicit SYSU219 experiment choices approved by the user.
    physics_step_s: float = 0.0025
    disturbance_start_iteration: int = 0
    disturbance_full_iteration: int = 14_999
    max_payload_kg: float = 6.0
    decimation: int = 4
    episode_length_s: float = 10.0
    is_finite_horizon: bool = True

    def __post_init__(self):
        self.sim.dt = self.physics_step_s
        self.sim.render_interval = self.decimation
        self.scene.robot = SYSU219_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = str(Path(__file__).with_name("sysu219_rough_contact.usda"))
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.49)
        motor = self.scene.robot.actuators["base_legs"]
        motor.class_type = dynamics.PaperDCMotor
        motor.stiffness = 55.0
        motor.damping = 0.8
        terrain = self.scene.terrain
        terrain.class_type = PaperTerrainImporter
        terrain.terrain_type = "generator"
        terrain.terrain_generator = TerrainGeneratorCfg(
            class_type=PaperFractalTerrain, size=(12.0, 12.0),
            num_rows=3, num_cols=3, curriculum=False, sub_terrains={},
        )
        terrain.max_init_terrain_level = None
        terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.41, 0.56, 0.52), roughness=0.95)
        self.sim.physics_material = terrain.physics_material
        self.scene.height_scanner = None
        self.scene.sky_light = None
        self.scene.contact_forces = None
        configure_paper_foot_sensors(self.scene)
        if carb.settings.get_settings().get("/app/window/enabled"):
            terrain.class_type = PaperReplayTerrainImporter
            self.viewer.origin_type = "world"
            self.scene.sky_light = AssetBaseCfg(
                prim_path="/World/PaperKeyLight",
                spawn=sim_utils.DistantLightCfg(intensity=1500),
                init_state=AssetBaseCfg.InitialStateCfg(rot=(0.8660254, 0.4, -0.3, 0)),
            )
            studio = get_extension_path_from_name("omni.kit.stage_templates")
            self.scene.paper_fill = AssetBaseCfg(
                prim_path="/World/PaperFill",
                spawn=sim_utils.DomeLightCfg(
                    intensity=0.7, texture_file=f"{studio}/data/CarLight_512x256.hdr", texture_format="latlong",
                ),
            )


@configclass
class SYSU219RMAStage1V1EnvCfg(SYSU219RMAPaperEnvCfg):
    """Contact-impulse and stance-referenced reward comparison for SYSU219."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ground_impact.func = _GroundImpulse
        self.rewards.action_magnitude.params = {"stance_relative": True}


@configclass
class SYSU219RMAStage1V2EnvCfg(SYSU219RMAStage1V1EnvCfg):
    """Increase the final payload range of the v1 teacher curriculum to 0–24 kg."""

    max_payload_kg: float = 24.0


class PaperRMAEnv(ManagerBasedRLEnv):
    def restore_paper_iteration(self, completed_updates: int):
        """Apply the restored physics curriculum to the robots already in the scene."""
        self.paper_iteration = completed_updates
        if hasattr(self, "_paper_dynamics"):
            self._paper_dynamics.sample(self, torch.arange(self.num_envs, device=self.device))

    def __init__(self, cfg: SYSU219RMAPaperEnvCfg, **kwargs):
        # Hydra overrides are applied after config __post_init__.
        if cfg.physics_step_s <= 0 or cfg.disturbance_full_iteration <= cfg.disturbance_start_iteration:
            raise ValueError("Require positive physics_step_s and an increasing disturbance curriculum.")
        substeps = round(0.01 / cfg.physics_step_s)
        if substeps < 1 or not math.isclose(substeps * cfg.physics_step_s, 0.01):
            raise ValueError("physics_step_s must divide the published 0.01 s control period exactly.")
        cfg.decimation = substeps
        cfg.sim.dt = cfg.physics_step_s
        cfg.sim.render_interval = substeps
        self.paper_iteration = 0
        self._initial_camera_set = False
        super().__init__(cfg, **kwargs)

    def reset(self, seed=None, env_ids=None, options=None):
        result = super().reset(seed=seed, env_ids=env_ids, options=options)
        if self.sim.has_gui() and not self._initial_camera_set:
            pos = self.scene["robot"].data.root_link_pos_w[0].detach().cpu().tolist()
            eye = (pos[0] - 2.3, pos[1] - 2.7, pos[2] + 1.35)
            self.sim.set_camera_view(eye=eye, target=pos)
            self._initial_camera_set = True
        return result
