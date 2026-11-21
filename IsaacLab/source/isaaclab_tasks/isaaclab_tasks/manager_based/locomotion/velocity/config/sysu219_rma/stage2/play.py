"""Watch the latest trained SYSU219 Stage 2 student at its deployment rates."""

import argparse
from pathlib import Path
import time


ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "IsaacLab").parent
LOG_ROOT = ROOT / "logs/rsl_rl/sysu219_rma_stage2"


def latest_checkpoint(log_root: Path) -> Path:
    # Run names are timestamps. A new run may only have its untrained student_0.
    for run in sorted(log_root.glob("*/"), reverse=True):
        candidates = []
        for path in run.glob("student_*.pt"):
            index = path.stem.removeprefix("student_")
            if index.isdecimal() and int(index) > 0:
                candidates.append((int(index), path))
        if candidates:
            return max(candidates)[1]
    raise FileNotFoundError(f"No trained student checkpoint yet in {log_root}")


def play(app, args, checkpoint_path):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.locomotion.velocity.config.sysu219_rma.stage2.learner import RolloutState
    from isaaclab_tasks.manager_based.locomotion.velocity.config.sysu219_rma.stage2.models import (
        AdaptationModule, FrozenTeacher, LATENT_DIM, STATE_DIM,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    cfg = parse_env_cfg(config["task"], device=args.device, num_envs=args.num_envs)
    cfg.seed = config["seed"]
    control = FrozenTeacher.from_checkpoint(ROOT / config["teacher_checkpoint"], cfg.sim.device)
    control.load_state_dict(checkpoint["frozen_teacher_state_dict"])
    student = AdaptationModule().to(cfg.sim.device).eval()
    student.load_state_dict(checkpoint["student_state_dict"])
    env = gym.make(config["task"], cfg=cfg)
    try:
        vec = RslRlVecEnvWrapper(env)
        raw = env.unwrapped
        raw.restore_paper_iteration(config["dynamics_iteration"])
        with torch.inference_mode():
            state = RolloutState(vec, control)
            obs, history, ages = state.observations, state.history, state.episode_steps
            stance = raw.scene["robot"].data.default_joint_pos
            latent = history.new_zeros((vec.num_envs, LATENT_DIM))
            step = 0
            # The 100 Hz controller holds each 10 Hz adaptation result for ten
            # control steps. A new episode always gets a fresh estimate immediately.
            while app.is_running() and (args.steps is None or step < args.steps):
                started = time.monotonic()
                refresh = ages.remainder(10) == 0
                if refresh.any():
                    latent[refresh] = student(history[refresh])
                policy = obs["policy"]
                action = control.action(policy, latent)
                if vec.clip_actions is not None:
                    action = action.clamp(-vec.clip_actions, vec.clip_actions)
                pair = control.normalized_history_pair(policy[:, :STATE_DIM], action)
                obs, _, done, _ = vec.step(action)
                done = done.bool()
                history = torch.cat((history[:, 1:], pair[:, None]), dim=1)
                if done.any():
                    padding = control.normalized_history_pair(obs["policy"][done, :STATE_DIM], stance[done])
                    history[done] = padding[:, None]
                ages = torch.where(done, 0, ages + 1)
                step += 1
                if args.real_time:
                    time.sleep(max(0.0, raw.step_dt - (time.monotonic() - started)))
        print(f"Played {step} control steps.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Override the latest trained student checkpoint.")
    parser.add_argument("--num_envs", "--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, help="Stop after this many control steps.")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or (args.steps is not None and args.steps < 1):
        parser.error("Environment count and step limit must be positive.")
    path = args.checkpoint.expanduser() if args.checkpoint else latest_checkpoint(LOG_ROOT)
    print(f"Playing Stage 2 student: {path}", flush=True)
    app = AppLauncher(args).app
    try:
        play(app, args, path)
    finally:
        app.close()
