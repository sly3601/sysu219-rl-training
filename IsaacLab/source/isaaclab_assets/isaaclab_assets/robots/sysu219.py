"""Configuration for SYSU219 quadruped robot."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


##
# Configuration
##

SYSU219_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(ISAACLAB_ASSETS_DATA_DIR) / "Robots" / "SYSU219" / "sysu219.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        # 真实站立高度调整完成
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.67,
            ".*_calf_joint": -1.30,
        },
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.9,

    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],

            # 单电机最大扭矩约 200 Nm， 保守写100Nm
            effort_limit=165.0,
            saturation_effort=200.0,

            # 先写 20 rad/s，3.18 转/秒
            velocity_limit=22.0,

            # PD 参数，结合已经调通的MPC
            stiffness={
                ".*_hip_joint": 285.0,
                ".*_thigh_joint": 265.0,
                ".*_calf_joint": 265.0,
            },
            damping={
                ".*_hip_joint": 4.5,
                ".*_thigh_joint": 4.5,
                ".*_calf_joint": 4.5,
            },

            # 被动关节摩擦和输出轴附加惯量沿用机器人 USD 中的辨识参数。
        ),
    },
)
"""Configuration of SYSU219 quadruped using DC motor model."""
