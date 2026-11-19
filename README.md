# SYSU219 强化学习工作区

## 直接训练

在终端激活 `isaaclab`，进入这个工作区的根目录。若当前终端已经激活该环境，直接执行 `cd` 一行即可。

```bash
conda activate isaaclab
cd ~/RL/rl_training
```

沿用免 JIT 的运行方式时，在当前终端先执行一次 `export PYTORCH_JIT=0`。这是 Python 运行设置。

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-SYSU219-v0 \
  --num_envs 1024 --max_iterations 3000 --headless
```

四个参数分别选择 SYSU219 平地任务、1024 个并行仿真环境、3000 轮 PPO 更新、无窗口模式。训练日志、检查点和当次配置快照保存在 `logs/rsl_rl/sysu219_flat/`。

## 边训练边看

省略 `--headless` 会打开 Isaac Sim 窗口。SYSU219 平地训练和回放都支持 Alt 加鼠标左键拖动绕观察点旋转，用滚轮缩放。回放时也可以直接拖动右键绕观察点旋转，用中键拖动平移，Shift 加右键拖动原地转头。要在已打开的窗口中解除跟随，在 IsaacLab 面板的 Viewer Settings 里把 Follow Mode 切到 Env。

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-SYSU219-v0 \
  --num_envs 1024 --max_iterations 3000
```

普通窗口默认使用 Hydra Storm。它通过 Fabric 读取 GPU 上每条腿和躯干的实时姿态。窗口隐藏碰撞体，只绘制机器人的外观网格，物理碰撞仍然生效。窗口模式还要刷新画面，训练速度会比无窗口模式慢。[窗口配置](IsaacLab/apps/isaaclab.python.pxr.kit)

## 默认值在哪里

| 之前写在命令里的选项 | 现在的默认值 | 配置位置 |
| --- | --- | --- |
| `--kit_args` 中的 `asset_root/cloud` | 无窗口仿真从工作区的 `assets/` 读取地面等资源 | [无窗口仿真配置](IsaacLab/apps/isaaclab.python.headless.kit) |
| `--kit_args` 中的 `registryEnabled=false` | 无窗口仿真使用已安装的 Kit 扩展缓存 | [无窗口仿真配置](IsaacLab/apps/isaaclab.python.headless.kit) |
| `env.commands.base_velocity.debug_vis=false` | 关闭速度箭头 | [平地任务配置](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/sysu219/flat_env_cfg.py) |
| `env.scene.sky_light=null` | 平地训练场景省略天空灯和 HDR 贴图 | [平地任务配置](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/sysu219/flat_env_cfg.py) |
| `env.events.base_external_force_torque=null` | 跳过原先每次重置时施加零力和零力矩的回调 | [平地任务配置](IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/sysu219/flat_env_cfg.py) |

无窗口仿真配置按 `.kit` 文件的位置解析 `assets/` 路径，供使用该配置的入口共用。以后换强化学习库时，使用 `IsaacLab/scripts/reinforcement_learning/` 下对应库的训练入口，并按该入口的参数运行。同一个 SYSU219 平地任务继续使用上表里的任务配置。

## COLCON_IGNORE

`COLCON_IGNORE` 是 ROS 2 的扫描标记，提示 `colcon` 跳过这个目录。独立运行 Isaac Lab 的 Python 训练与它无关。
