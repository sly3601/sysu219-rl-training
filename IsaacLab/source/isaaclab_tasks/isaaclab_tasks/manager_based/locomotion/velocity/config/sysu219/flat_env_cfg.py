# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import carb
from isaacsim.core.utils.extensions import get_extension_path_from_name
from pxr import UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass

from .rough_env_cfg import SYSU219RoughEnvCfg

# 新增
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class SYSU219FlatEnvCfg(SYSU219RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Keep this task usable with the local simulation assets.
        self.commands.base_velocity.debug_vis = False
        self.scene.sky_light = None
        settings = carb.settings.get_settings()
        if settings.get("/app/window/enabled") and settings.get("/renderer/active") == "pxr":
            # Storm reads live GPU link poses through the Fabric scene delegate.
            self.sim.use_fabric = True
            self._setup_robot_view()
        # The inherited reset callback applies zero force and torque.
        self.events.base_external_force_torque = None

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # 原地踏步
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # 目标速度范围
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # 惩罚足端接触地面时的水平滑动
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # 电机扭矩过大惩罚
        self.rewards.dof_torques_l2.weight = -0.0001

        self.rewards.track_lin_vel_xy_exp.weight = 2.5  # xy 线速度跟踪奖励权重

        self.rewards.lin_vel_z_l2.weight = -3.7          # 机身 z 方向速度惩罚
        self.rewards.ang_vel_xy_l2.weight = -0.1         # 机身 roll/pitch 角速度惩罚
        self.rewards.flat_orientation_l2.weight = -1.0   # 机身姿态倾斜惩罚；越负越希望身体保持水平

        self.rewards.action_rate_l2.weight = -0.02  # 动作变化率惩罚
        self.rewards.dof_acc_l2.weight = -5.0e-7    # 关节加速度惩罚

        # 目标速度小于0.4时，要限制髋关节偏离默认角度。
        self.rewards.stand_still_hip_deviation = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.8,
            params={
                "command_name": "base_velocity",
                "command_threshold": 1.0,  # 目标速度小于1.0时，才会惩罚髋关节偏离默认角度
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            },
)





        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

    def _setup_robot_view(self):
        self.viewer.origin_type = "env"
        self.viewer.eye = (2.6, 2.6, 1.7)
        self.viewer.lookat = (0.0, 0.0, 0.35)
        self.scene.sky_light = AssetBaseCfg(
            prim_path="/World/ReplayLight",
            spawn=sim_utils.DistantLightCfg(intensity=1500.0),
        )


class _PlayTerrainImporter(TerrainImporter):
    def import_ground_plane(self, name: str, size: tuple[float, float] = (100.0, 100.0)):
        # The default 2,000 km visual mesh occludes the robots in Storm. PhysX uses a separate infinite plane.
        super().import_ground_plane(name, size)
        stage = sim_utils.get_current_stage()
        ground = f"{self.cfg.prim_path}/{name}"
        material = UsdShade.Material.Define(stage, f"{ground}/Looks/PlayGrid")
        material.GetPrim().GetReferences().AddReference(f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/sysu219_play_grid.usda")
        floor = stage.GetPrimAtPath(f"{ground}/Environment/Geometry")
        UsdShade.MaterialBindingAPI.Apply(floor).Bind(material)


class SYSU219FlatEnvCfg_PLAY(SYSU219FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        if self.scene.sky_light is None:
            self._setup_robot_view()
        settings = carb.settings.get_settings()
        if settings.get("/app/window/enabled") and settings.get("/renderer/active") == "pxr":
            self.scene.terrain.class_type = _PlayTerrainImporter
            self.viewer.origin_type = "world"
            self.viewer.eye = (8.5, -3.4, 2.1)
            self.viewer.lookat = (1.3, 0.0, 0.25)
            studio = get_extension_path_from_name("omni.kit.stage_templates")
            self.scene.replay_fill = AssetBaseCfg(
                prim_path="/World/ReplayFill",
                spawn=sim_utils.DomeLightCfg(
                    intensity=0.4, texture_file=f"{studio}/data/CarLight_512x256.hdr", texture_format="latlong"
                ),
            )
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
