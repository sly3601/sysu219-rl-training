"""Train the SYSU219 RMA history encoder against the frozen Stage 1 v1 actor."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback


ROOT = next(p for p in Path(__file__).resolve().parents if p.name == "IsaacLab").parent
TEACHER = ROOT / "logs/rsl_rl/sysu219_rma_stage1_v1/2026-11-20_13-32-50/model_14999.pt"
LOG_ROOT = ROOT / "logs/rsl_rl/sysu219_rma_stage2"
TASK = "Isaac-RMA-SYSU219-Stage1-v1"


def parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4000)
    parser.add_argument("--steps-per-env", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT, help="Parent directory for this run.")
    parser.add_argument("--resume", type=Path, metavar="STUDENT_PT",
                        help="Continue the same run from its most recent saved student.")
    parser.add_argument("--save-every", type=int, default=50)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.steps_per_env, args.iterations, args.save_every) < 1:
        parser.error("Environment count, steps, iterations and save interval must be positive.")
    if args.num_envs * args.steps_per_env % 4:
        parser.error("The rollout size must be divisible by the four mini-batches.")
    if not TEACHER.is_file():
        parser.error(f"Missing Stage 1 teacher: {TEACHER}")
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            parser.error(f"Missing Stage 2 checkpoint: {args.resume}")
    return args


def save_checkpoint(run_dir, iteration, student, teacher, optimizer, config):
    import torch

    path = run_dir / f"student_{iteration}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1,
        "iteration": iteration,
        "samples": iteration * config["samples_per_iteration"],
        "student_state_dict": student.state_dict(),
        "frozen_teacher_state_dict": teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "teacher_checkpoint": config["teacher_checkpoint"],
        "teacher_sha256": config["teacher_sha256"],
        "config": config,
    }, temporary)
    temporary.replace(path)


def load_resume(path, config):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1 or checkpoint["config"] != config:
        raise ValueError(f"Checkpoint training settings do not match this run: {path}")
    if json.loads((path.parent / "config.json").read_text()) != config:
        raise ValueError(f"Run configuration and checkpoint disagree: {path.parent}")
    if any(checkpoint[key] != config[key] for key in ("teacher_checkpoint", "teacher_sha256")):
        raise ValueError(f"Checkpoint teacher does not match the configured teacher: {path}")
    iteration = checkpoint["iteration"]
    if not 0 <= iteration < config["iterations"] or checkpoint["samples"] != iteration * config["samples_per_iteration"]:
        raise ValueError(f"Invalid or already finished training iteration in {path}")
    saved = [p for p in path.parent.glob("student_*.pt") if p.stem[8:].isdigit()]
    if path.name != f"student_{iteration}.pt" or any(int(p.stem[8:]) > iteration for p in saved):
        raise ValueError(f"Specify the latest saved checkpoint for this run: {path.parent}")
    return checkpoint


def train(args):
    import gymnasium as gym
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.manager_based.locomotion.velocity.config.sysu219_rma.stage2.learner import (
        RolloutState, run_iteration,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.sysu219_rma.stage2.models import (
        AdaptationModule, FrozenTeacher,
    )

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    samples = args.num_envs * args.steps_per_env
    config = {
        "task": TASK, "teacher_checkpoint": str(TEACHER.relative_to(ROOT)),
        "teacher_sha256": hashlib.sha256(TEACHER.read_bytes()).hexdigest(),
        "seed": args.seed, "num_envs": args.num_envs, "steps_per_env": args.steps_per_env,
        "samples_per_iteration": samples, "iterations": args.iterations, "minibatches": 4,
        "learning_rate": 5e-4, "history_steps": 50, "control_hz": 100,
        "sampling_mode": "continuous", "episode_length_s": cfg.episode_length_s,
        "dynamics_iteration": cfg.disturbance_full_iteration,
    }
    checkpoint = None
    start_iteration = 0
    if args.resume is None:
        run_dir = args.log_root.expanduser().resolve() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        run_dir.mkdir(parents=True)
    else:
        checkpoint = load_resume(args.resume, config)
        run_dir = args.resume.parent
        start_iteration = checkpoint["iteration"]
    cfg.log_dir = str(run_dir)
    teacher = FrozenTeacher.from_checkpoint(TEACHER, cfg.sim.device)
    env = gym.make(TASK, cfg=cfg)
    try:
        vec = RslRlVecEnvWrapper(env)
        env.unwrapped.restore_paper_iteration(cfg.disturbance_full_iteration)
        # Independent of the amount of randomness consumed while creating the scene.
        torch.manual_seed(args.seed)
        student = AdaptationModule().to(vec.device)
        optimizer = torch.optim.Adam(student.parameters(), lr=config["learning_rate"])
        rollout_state = RolloutState(vec, teacher)
        print(f"Frozen teacher: {TEACHER}\nStage 2 run: {run_dir}", flush=True)
        if checkpoint is None:
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            save_checkpoint(run_dir, 0, student, teacher, optimizer, config)
        else:
            student.load_state_dict(checkpoint["student_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Restored student and Adam at iteration {start_iteration}. "
                  "The simulator and per-robot histories start fresh after relaunch.", flush=True)
        purge_step = start_iteration + 1 if checkpoint is not None else None
        with SummaryWriter(str(run_dir), flush_secs=10, purge_step=purge_step) as writer:
            for iteration in range(start_iteration + 1, args.iterations + 1):
                metrics = run_iteration(vec, teacher, student, optimizer, args.steps_per_env, rollout_state)
                for name, value in metrics.items():
                    writer.add_scalar(f"Stage2/{name}", value, iteration)
                print(f"[{iteration}/{args.iterations}] samples={samples * iteration}  "
                      f"on-policy MSE={metrics['mse']:.5g}  "
                      f"forward={metrics['mean_forward_speed']:.3f} m/s  "
                      f"full_history={metrics['full_history_fraction']:.1%}", flush=True)
                if iteration % args.save_every == 0 or iteration == args.iterations:
                    save_checkpoint(run_dir, iteration, student, teacher, optimizer, config)
                    writer.flush()
    finally:
        env.close()


if __name__ == "__main__":
    from isaaclab.app import AppLauncher

    arguments = parse_args()
    app = AppLauncher(arguments).app
    try:
        train(arguments)
    except BaseException:
        # Kit's native shutdown can exit before Python displays an uncaught exception.
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        app.close()
