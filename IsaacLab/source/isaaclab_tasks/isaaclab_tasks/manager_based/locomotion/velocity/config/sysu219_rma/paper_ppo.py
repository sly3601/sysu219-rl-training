"""RSS 2021 RMA Phase 1, on top of the installed RSL-RL rollout and GAE implementation."""

import math
from itertools import chain

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class ObservationMoments(nn.Module):
    """Online observation scaling from the authors' publicly linked RaisimGymVecEnv."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(size, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1.0e-4, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = (x - self.mean.to(x.dtype)) * torch.rsqrt(self.variance.to(x.dtype) + 1.0e-8)
        return scaled.clamp(-10.0, 10.0)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, raw: torch.Tensor, distributed: bool):
        x = raw.double()
        n = self.count.new_tensor(x.size(0))
        if distributed:
            totals = torch.cat((n.reshape(1), x.sum(0), x.square().sum(0)))
            torch.distributed.all_reduce(totals)
            n = totals[0]
            sums, squared_sums = totals[1:].chunk(2)
            batch_mean = sums / n
            batch_var = (squared_sums / n - batch_mean.square()).clamp_min(0)
        else:
            batch_var, batch_mean = torch.var_mean(x, dim=0, unbiased=False)
        delta = batch_mean - self.mean
        weight = n / (self.count + n)
        # Parallel variance merge, including the initial count/variance prior.
        self.variance.add_(weight * (batch_var - self.variance + delta.square() * (1 - weight)))
        self.mean.add_(weight * delta)
        self.count.add_(n)


class _EncodedPolicyInput(nn.Module):
    def __init__(self, head: MLP, activation: str):
        super().__init__()
        self.encoder = MLP(17, 8, [256, 128], activation, last_activation=activation)
        self.head = head

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        state_and_previous_action = observation[..., :42]
        extrinsics = self.encoder(observation[..., 42:])
        return self.head(torch.cat((state_and_previous_action, extrinsics), dim=-1))


class PaperModel(MLPModel):
    """The policy and critic have independent encoders. The policy's encoder supplies Phase 2 labels."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, *, activation, obs_normalization=True, **kwargs):
        if obs_groups[obs_set] != ["policy", "privileged"]:
            raise ValueError("Expected observations ordered as [policy, privileged].")
        if obs["policy"].shape[-1] != 42 or obs["privileged"].shape[-1] != 17:
            raise ValueError("RMA Phase 1 expects 30 state + 12 previous-action inputs and 17 environment inputs.")
        if not obs_normalization:
            raise ValueError("The paper task uses shared online observation normalization.")
        super().__init__(obs, obs_groups, obs_set, output_dim, activation=activation, obs_normalization=True, **kwargs)
        self.obs_normalizer = ObservationMoments(self.obs_dim)
        self.mlp = _EncodedPolicyInput(self.mlp, activation)
        # Orthogonal weights and unchanged torch Linear biases match the authors' linked training code.
        for layer in self.mlp.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))

    def _get_latent_dim(self):
        return 42 + 8

    def update_normalization(self, obs):
        """PaperPPO updates the shared moments before sampling, including the final bootstrap observation."""


class PaperPPO(PPO):
    """PPO with value clipping relative to the old value and a lower bound on action noise."""

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str):
        algorithm = PPO.construct_algorithm(obs, env, cfg, device)
        algorithm.critic.obs_normalizer = algorithm.actor.obs_normalizer
        algorithm.vec_env = env
        algorithm.task = env.unwrapped
        algorithm.task.paper_iteration = 0
        algorithm._needs_initial_reset = True
        return algorithm

    def act(self, obs: TensorDict) -> torch.Tensor:
        first_action = self._needs_initial_reset
        if first_action:
            # The runner randomizes episode ages on startup. Begin the new scene at step zero.
            obs, _ = self.vec_env.reset()
            obs = obs.to(self.device)
            self._needs_initial_reset = False
        raw = torch.cat((obs["policy"], obs["privileged"]), dim=-1)
        moments = self.actor.obs_normalizer
        # At a continuing rollout boundary compute_returns already counted this observation.
        if first_action or self.storage.step > 0:
            moments.update(raw, self.is_multi_gpu)
        # Both networks and the rollout buffer use the same normalized observation.
        prepared = moments(raw)
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.actor.distribution.update(self.actor.mlp(prepared))
        self.transition.actions = self.actor.distribution.sample().detach()
        self.transition.values = self.critic.mlp(prepared).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(param.detach() for param in self.actor.output_distribution_params)
        policy, privileged = prepared.split((42, 17), dim=-1)
        self.transition.observations = TensorDict(
            {"policy": policy, "privileged": privileged}, batch_size=obs.batch_size,
        )
        return self.transition.actions

    def compute_returns(self, obs):
        raw = torch.cat((obs["policy"], obs["privileged"]), dim=-1)
        self.actor.obs_normalizer.update(raw, self.is_multi_gpu)
        super().compute_returns(obs)

    def update(self) -> dict[str, float]:
        value_total = torch.zeros((), device=self.device)
        policy_total = torch.zeros_like(value_total)
        entropy_total = torch.zeros_like(value_total)
        params = list(chain(self.actor.parameters(), self.critic.parameters()))
        batches = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in batches:
            prepared = torch.cat((batch.observations["policy"], batch.observations["privileged"]), dim=-1)
            self.actor.distribution.update(self.actor.mlp(prepared))
            new_log_prob = self.actor.get_output_log_prob(batch.actions)
            ratio = (new_log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
            advantages = batch.advantages.squeeze(-1)
            clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()

            value = self.critic.mlp(prepared)
            # Supplement §II sets bounds relative to the old value, which can be negative.
            radius = self.clip_param * batch.values.abs()
            clipped_value = batch.values + (value - batch.values).clamp(-radius, radius)
            value_loss = torch.maximum((value - batch.returns).square(), (clipped_value - batch.returns).square()).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                # Project after Adam so sampling and log likelihood always use the very same constrained std.
                std = self.actor.distribution.std_param
                floor = torch.nextafter(std.new_tensor(0.2), std.new_tensor(math.inf))
                std.clamp_min_(floor)
                value_total += value_loss.detach()
                policy_total += surrogate_loss.detach()
                entropy_total += self.actor.output_entropy.mean()

        self.storage.clear()
        self.task.paper_iteration += 1
        count = self.num_learning_epochs * self.num_mini_batches
        return {
            "value": (value_total / count).item(),
            "surrogate": (policy_total / count).item(),
            "entropy": (entropy_total / count).item(),
        }

    def save(self) -> dict:
        checkpoint = super().save()
        checkpoint["paper_completed_updates"] = self.task.paper_iteration
        return checkpoint

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        restore_iteration = super().load(loaded_dict, load_cfg, strict)
        if restore_iteration:
            self._needs_initial_reset = True
            self.task.restore_paper_iteration(int(loaded_dict["paper_completed_updates"]))
            # RSL-RL 5.0.1 stores the most recent zero-based iteration; learn() needs the NEXT index.
            loaded_dict["iter"] = self.task.paper_iteration
        return restore_iteration
