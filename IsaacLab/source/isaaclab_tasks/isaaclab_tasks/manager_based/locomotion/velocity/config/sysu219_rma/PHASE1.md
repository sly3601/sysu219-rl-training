# SYSU219 上的 RMA 第一阶段

从 `~/RL/rl_training` 运行，使用已激活的 `isaaclab` 环境。该任务训练基础策略和环境编码器。

| 任务 | 足地冲击使用的量 | 动作幅值惩罚使用的量 | 检查点与训练曲线目录 |
|---|---|---|---|
| `Isaac-RMA-SYSU219-Stage1-v2` | 与 v1 相同 | 与 v1 相同 | `logs/rsl_rl/sysu219_rma_stage1_v2/` |
| `Isaac-RMA-SYSU219-Stage1-v1` | 最后一个物理子步的单脚净冲量（N·s） | 相对站姿的目标偏移，除以 0.4 rad | `logs/rsl_rl/sysu219_rma_stage1_v1/` |
| `Isaac-RMA-SYSU219-Stage1-v0` | 单脚地面合力（N） | 绝对目标角（rad） | `logs/rsl_rl/sysu219_rma_paper_stage1/` |

v1 是在 SYSU219 上验证奖励的对照实验。足地冲击权重为 −0.02，动作幅值权重为 −0.07。两版策略均输出绝对关节角并交给 PD 控制器。要在 v1 里单独测冲量改动，可给训练命令追加 `env.rewards.action_magnitude.params.stance_relative=false`，并用 `agent.experiment_name=sysu219_rma_impulse_only` 单独保存对照曲线。

作者后续实现在每只脚上累加接触点冲量的模长。PhysX 分别提供法向接触力和切向摩擦力，其接口没有给出两类接触点的一一对应关系。v1 先合成单脚接触力，再乘物理步长（默认 0.0025 秒）并取模。这在单个接触点时与作者做法相同，多点受力方向不同时会得到更小的数值。动作幅值的站姿零点和 0.4 rad 尺度也取自作者后续代码。2021 年实验这两处的具体实现未能核实。[足底接触](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp#L1032-L1043) · [动作尺度](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp#L183-L191)

## v2 重载训练

在 `~/RL/rl_training` 激活 `isaaclab` 环境后，运行 `python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-RMA-SYSU219-Stage1-v2 --headless`。

v2 只把躯干附加质量的训练上限从 6 kg 提高到 24 kg。沿用 v1 的线性课程，从标称负载开始，到第 15000 轮使用完整的 0–24 kg 范围。PPO 每轮仍采 4096 × 20 步。下文记录的 0–6 kg 训练范围适用于 v0 和 v1。

正式配对测试继续使用 0–24 kg 和每控制步 1% 的扰动重采样概率。测试沿用此前重载评估的其余参数、地图种子、起点和扰动日程，教师使用本次 v2 训练选定的检查点。训练中的重采样概率仍是 0.4%。

## v1 奖励验证（2026-11-20）

在同一地形种子、16 个相同初始状态、标称动力学下回放，使用后期奖励系数。下表统计第一次摔倒或满 10 秒前的行为。折扣回报采用 PPO 的 γ=0.998，数值越高越好。经典 PPO 使用原有 50 Hz 观测和 `model_2999.pt`。做 PD 适配时，折算目标角以接近原控制器要求的力矩，RMA 环境仍使用相同的电机参数。

| 控制方式 | 平均回合时长 | 平均向前位移 | 平均折扣回报 |
|---|---:|---:|---:|
| 经典 PPO，适配 PD | 6.95 秒 | +0.318 m | −183.75 |
| 经典 PPO，直接使用目标角 | 0.30 秒 | −0.001 m | −47.03 |
| 旧 RMA 第 10750 轮 | 0.06 秒 | +0.009 m | −255.42 |
| 默认站姿 | 9.39 秒 | −0.220 m | −532.58 |

行走策略在 9/16 个起点超过旧 RMA 的快倒策略。另一种 0.30 秒就倒的控制方式得分更高，说明仍有早倒得利的情况。适配步态的有效足地接触抽样中，62.5% 有多个法向接触点。

另用 256 个环境从零训练 180 轮，每个环境每轮 20 步。在另外 32 只机器人的固定起点上独立回放，编号 0、100、179 的检查点平均存活 0.278、0.209、0.185 秒，平均向前位移 −0.132、−0.111、−0.105 m。最后一个检查点的 32 只全部向后移动并摔倒。正式训练尚未通过验证。检查点、TensorBoard 曲线和逐起点的 `evaluation.json` 位于 `logs/smoke_tests/sysu219_rma_stage1_v1/2026-11-20_13-24-03/`。

## 来自 2021 年论文的设置

| 项目 | 实现 |
|---|---|
| 策略输入输出 | 12 个关节角、12 个关节角速度、roll/pitch、4 个接触位组成 30 维状态。拼接上一动作和 8 维环境编码，输出 12 个目标关节角。关节顺序使用本机 SYSU219 的 articulation 顺序。 |
| 环境编码 | 输入为额外负载、质心的两个水平位移、12 个电机强度、足地摩擦、局部地形高度。四足下方实际碰撞网格的世界高度按 0.1 m 量化后取最大值。编码器为 17→256→128→8，策略为 50→128→128→12。 |
| 物理范围 | 负载 0 至 6 kg；质心偏移按表 I 的厘米换算为每个水平轴 ±0.0015 m；电机强度 0.90 至 1.10；摩擦 0.05 至 4.5；Kp 50 至 60，Kd 0.4 至 0.8。每个控制步按 0.004 的概率重新采样。 |
| 地形与终止 | 固定难度分形地形，2 个倍频程，频率倍率 2，振幅倍率 0.25，z-scale 0.27。每回合最多 1000 步；机身离地低于 0.28 m、绝对 roll 超过 0.4 rad 或绝对 pitch 超过 0.2 rad 时终止。 |
| 奖励 | `paper_mdp.py` 依次实现向前、侧移与偏航、做功、足地冲击、扭矩变化、动作幅度、关节速度、姿态、竖向速度和足滑。向前速度上限 0.35 m/s。第 9 项依原文公式使用竖向速度平方。十项系数依次为 20、21、0.002、0.02、0.001、0.07、0.002、1.5、2、0.8。后九项按公式取负，第 3 至 10 项另乘从 0.03 开始、每次更新取前值的 0.997 次方的课程系数。奖励以每个控制步为单位。 |
| PPO | 原文每轮 80,000 条转移、4 个 minibatch、4 个 epoch、15,000 轮。固定 Adam 学习率 0.0005，γ=0.998、λ=0.95、策略裁剪 0.2、价值损失系数 0.5、熵系数 0，高斯标准差大于 0.2。价值裁剪按补充材料相对于旧价值的 20% 幅度计算。编码器和策略共同更新。 |

来源为论文第 III–IV 节、表 I，以及补充材料第 II 节、算法 1。[论文](https://ashish-kmr.github.io/rma-legged-robots/rma-locomotion-final.pdf) · [补充材料](https://ashish-kmr.github.io/rma-legged-robots/rma-locomotion-supplementary.pdf)

## 本实验的采样与重置

本机采用 4096 个环境，每只采 20 步（0.2 秒），每次 PPO 更新使用 81,920 条转移，比论文的每轮采样量多 2.4%。更新时，存活的环境保留姿态、速度、上一动作和回合计数，从下一控制步继续。只在摔倒或达到 1000 个控制步（10 秒）时重置对应机器人。批末 GAE 对存活环境使用价值估计，日志按实际结束的回合统计。这个跨轮连续采样约定与论文补充材料算法 1 的轮首重置不同。默认训练命令从零开始。显式续训时，检查点恢复模型、优化器、归一化和课程进度，新仿真场景从回合第零步开始。

## 在这台机器上如何补齐未公布细节

- 论文同时给出 100 Hz 控制与 0.025 s 物理步长。按你确认的配置，物理步长取作者后续公开值 0.0025 s，每 4 个物理步控制一次。论文只给出每轮样本总量，本实验在一张 RTX 4060 Ti 上运行，采用上节的并行采样配置。扰动范围从第 1 轮的标称值线性扩大，到第 15000 轮达到表 I 范围，起止轮次是本实验选择。质量、摩擦、电机和 PD 增益在各自范围内均匀采样。[后续公开配置](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/envs/rsg_a1_task/cfg.yaml)
- LeakyReLU、正交权重初始化、初始探索标准差 0.3、梯度范数上限 0.5 和独立价值网络参考作者主页链接的后续实现。训练的高斯策略以绝对弧度为单位使用线性均值输出。论文给出了目标角的物理意义；2021 训练时的网络末层和动作归一化细节没有披露。电机强度扰动按后续源码作用于目标相对站姿的偏移，然后使用 SYSU219 的执行器限制计算扭矩。[训练入口](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/envs/rsg_a1_task/runner.py) · [网络](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/algo/ppo/module.py) · [PPO](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/algo/ppo/ppo.py) · [执行器](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp#L977-L988)
- 网络输入采用在线均值与方差归一化，裁剪到 ±10；分母方差项加 1e-8。采样时记录当步归一化后的观测，PPO 重算概率时读同一份数据。推理使用检查点中的统计量。[作者后续观测处理](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/RaisimGymVecEnv.py#L122-L144)
- 地形沿用后续地形函数的基频 10 和每 12 m 取 200 个高度样点。Isaac Lab 在启动时生成一张 36 m × 36 m 的连续碰撞网格，每次 reset 在中心 24 m × 24 m 内重新抽取起点。不同地形种子产生不同地图；NumPy 和 RaiSim 的排列生成器不同。机器人使用 SYSU219 的站姿、零初速和已有足端碰撞形状。[作者地形函数](https://github.com/antonilo/rl_locomotion/blob/f71db8195bfb49b6d964bf53be3c5adadb0c11b1/raisimGymTorch/env/envs/rsg_a1_task/Environment.hpp#L672-L691) · [RaiSim 地形公式](https://github.com/raisimTech/raisimLib/blob/f0bb440762c09a9cc93cf6ad3a7f8552c6a4f858/raisim/linux/include/raisim/Terrain.hpp)

## 跨轮验证

此前使用同一检查点对照过 4096 × 20 和 2048 × 40。每轮均为 81,920 个样本。在 16 GiB RTX 4060 Ti 上完成预热后，四个 PPO 更新分别平均用时 3.64 秒和 3.94 秒。

在本机从 `model_100.pt` 恢复，用当时的 1000 × 80 配置连续运行两个完整 PPO 更新。第 80 步结束时有 959 只机器人存活。更新前后，机器人的位姿、关节、动作历史、回合计数与下一动作使用的原始观测逐项一致。最长连续回合达到本次测试的全部 160 步。另将一组回合时钟置于 998 和 999 步，验证了每只机器人在自己的第 1000 步重置。

## 早期配置的小规模验证

8 个 GPU 仿真环境，每环境每轮 1000 个控制步，3 轮合计 24,000 条转移。检查点记录了优化器、课程轮次和归一化统计量。用检查点回放了 100 个控制步。

第 3 次更新的日志里，最近回合平均约 26 个控制步（0.26 秒）；足地冲击是负奖励中绝对值最大的一项。该验证覆盖训练链路，步态和论文的性能指标需要训练完成后独立测量。
