"""On-policy RMA latent regression with continuous, per-episode state/action histories."""

import torch
from torch.nn import functional as F

from .models import HISTORY_LENGTH, LATENT_DIM, PROPRIO_DIM, STATE_DIM


class RolloutState:
    """Live state immediately before the next decision, retained across Adam updates."""

    @torch.no_grad()
    def __init__(self, env, teacher):
        self.observations, _ = env.reset()
        stance = env.unwrapped.scene["robot"].data.default_joint_pos
        padding = teacher.normalized_history_pair(self.observations["policy"][:, :STATE_DIM], stance)
        self.history = padding[:, None, :].expand(-1, HISTORY_LENGTH, -1)
        self.episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=stance.device)


class RolloutHistory:
    """Gather causal windows for both inference and the shuffled supervised updates.

    Rows contain the preceding 50 frames, the current rollout's measured pairs,
    and one possible reset-padding row per decision (including the next decision).
    """

    def __init__(self, steps: int, prefix: torch.Tensor):
        num_envs = prefix.shape[0]
        device = prefix.device
        self.padding_offset = HISTORY_LENGTH + steps
        self.rows = prefix.new_empty((self.padding_offset + steps + 1, num_envs, PROPRIO_DIM))
        self.rows[:HISTORY_LENGTH].copy_(prefix.transpose(0, 1))
        self.starts = torch.empty((steps + 1, num_envs), dtype=torch.long, device=device)
        self.offsets = torch.arange(-HISTORY_LENGTH, 0, device=device)

    def window(self, steps: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        frame_steps = steps[:, None] + self.offsets
        start = self.starts[steps, env_ids, None]
        # For episodes already in progress the prefix includes any needed padding.
        pad = (start >= 0) & (frame_steps < start)
        row_ids = torch.where(pad, self.padding_offset + start, HISTORY_LENGTH + frame_steps)
        return self.rows[row_ids, env_ids[:, None]]


@torch.no_grad()
def _collect(env, teacher, student, steps_per_env, history, targets, carry):
    raw = env.unwrapped
    observations = carry.observations
    robot = raw.scene["robot"]
    standing_action = robot.data.default_joint_pos
    device = standing_action.device
    env_ids = torch.arange(env.num_envs, device=device)
    episode_start = -carry.episode_steps
    fell_during_rollout = torch.zeros_like(env_ids, dtype=torch.bool)
    # Accumulate on-device. Only scalar logging at the end of an iteration crosses to CPU.
    latent_error, total_reward, total_speed, ended, ended_steps, timeouts = torch.zeros(6, device=device).unbind()
    for t in range(steps_per_env):
        policy_obs = observations["policy"]
        state = policy_obs[:, :STATE_DIM]
        history.starts[t].copy_(episode_start)
        history.rows[history.padding_offset + t].copy_(teacher.normalized_history_pair(state, standing_action))
        recent = history.window(torch.full_like(env_ids, t), env_ids)
        estimate = student(recent)
        target = teacher.latent(observations["privileged"])
        action = teacher.action(policy_obs, estimate)
        if env.clip_actions is not None:
            action = action.clamp(-env.clip_actions, env.clip_actions)
        history.rows[HISTORY_LENGTH + t].copy_(teacher.normalized_history_pair(state, action))
        targets[t].copy_(target)
        latent_error += (estimate - target).square().sum()
        total_speed += robot.data.root_link_lin_vel_b[:, 0].sum()
        observations, rewards, dones, _ = env.step(action)
        done = dones.bool()
        total_reward += rewards.sum()
        ended += done.sum()
        ended_steps += ((t + 1 - episode_start) * done).sum()
        fell_during_rollout |= raw.reset_terminated
        # A simultaneous fall and time limit counts as a fall for the success metric.
        timeouts += (raw.reset_time_outs & ~raw.reset_terminated).sum()
        episode_start = torch.where(done, t + 1, episode_start)

    # A dog may have reset on the last step. env.step already returned its new
    # initial observation, which supplies the padding for the next iteration.
    carry.observations = observations
    carry.episode_steps = steps_per_env - episode_start
    history.starts[steps_per_env].copy_(episode_start)
    history.rows[history.padding_offset + steps_per_env].copy_(
        teacher.normalized_history_pair(observations["policy"][:, :STATE_DIM], standing_action)
    )
    carry.history = history.window(torch.full_like(env_ids, steps_per_env), env_ids)

    samples = steps_per_env * env.num_envs
    step_numbers = torch.arange(steps_per_env, device=device)[:, None]
    complete_histories = (step_numbers - history.starts[:steps_per_env] >= HISTORY_LENGTH).float().mean()
    metrics = {
        "mse": latent_error / (samples * LATENT_DIM),
        "samples": samples,
        "mean_step_reward": total_reward / samples,
        "mean_forward_speed": total_speed / samples,
        "ended_episodes": ended,
        "rollout_no_fall_fraction": (~fell_during_rollout).float().mean(),
        "full_history_fraction": complete_histories,
    }
    if ended.item() > 0:
        metrics["mean_ended_episode_length"] = ended_steps / ended
        metrics["timeouts"] = timeouts
        metrics["timeout_fraction"] = timeouts / ended
    return metrics


def run_iteration(env, teacher, student, optimizer, steps_per_env: int, carry: RolloutState) -> dict[str, float]:
    """Collect a fresh batch using student predictions, then fit four shuffled mini-batches."""
    num_envs = env.num_envs
    samples = steps_per_env * num_envs
    if steps_per_env < 1 or num_envs < 1 or samples % 4:
        raise ValueError("Phase 2 needs a positive rollout with four equal mini-batches.")
    device = next(student.parameters()).device
    history = RolloutHistory(steps_per_env, carry.history)
    targets = torch.empty((steps_per_env, num_envs, LATENT_DIM), device=device)

    student.eval()
    metrics = _collect(env, teacher, student, steps_per_env, history, targets, carry)
    # Best single constant vector for these labels. Compare to rollout MSE
    # to see whether history predicts variation beyond the mean encoding.
    metrics["batch_constant_mse"] = targets.var(dim=(0, 1), unbiased=False).mean()
    if not torch.isfinite(metrics["mse"]):
        raise FloatingPointError("Non-finite student prediction or teacher label during rollout.")
    order = torch.randperm(samples, device=device)
    minibatch_size = samples // 4
    optimization_error = torch.zeros((), device=device)
    student.train()
    for minibatch in order.split(minibatch_size):
        optimizer.zero_grad(set_to_none=True)
        # Bound activation memory for the 50-frame CNN. Accumulate the exact
        # mini-batch mean before taking a single Adam step.
        for chunk in minibatch.split(1024):
            steps = chunk // num_envs
            env_ids = chunk % num_envs
            estimate = student(history.window(steps, env_ids))
            error = F.mse_loss(estimate, targets[steps, env_ids], reduction="sum")
            (error / (minibatch_size * LATENT_DIM)).backward()
            optimization_error += error.detach()
        optimizer.step()
    metrics["optimization_mse"] = optimization_error / (samples * LATENT_DIM)
    return {name: float(value) for name, value in metrics.items()}
