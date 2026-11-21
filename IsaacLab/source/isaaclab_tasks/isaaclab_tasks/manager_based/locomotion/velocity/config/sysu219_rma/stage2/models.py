"""Frozen RMA teacher and the temporal adaptation network from RMA §III-B/IV-B."""

from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from ..paper_ppo import PaperModel


STATE_DIM = 30
ACTION_DIM = 12
PROPRIO_DIM = STATE_DIM + ACTION_DIM
PRIVILEGED_DIM = 17
LATENT_DIM = 8
HISTORY_LENGTH = 50


class FrozenTeacher(nn.Module):
    """Expose the two deterministic parts of the trained actor using its frozen moments."""

    def __init__(self, actor: PaperModel):
        super().__init__()
        self.encoder = actor.mlp.encoder
        self.head = actor.mlp.head
        self.register_buffer("mean", actor.obs_normalizer.mean.float())
        self.register_buffer("inv_std", torch.rsqrt(actor.obs_normalizer.variance.float() + 1.0e-8))
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device) -> "FrozenTeacher":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["actor_state_dict"]
        if not all(torch.isfinite(value).all() for value in state_dict.values()):
            raise ValueError(f"Stage 1 actor contains non-finite weights or observation moments: {path}")
        if (state_dict["obs_normalizer.variance"] < 0).any() or state_dict["obs_normalizer.count"].item() <= 0:
            raise ValueError(f"Invalid Stage 1 observation moments: {path}")
        obs = TensorDict(
            {"policy": torch.zeros(1, PROPRIO_DIM), "privileged": torch.zeros(1, PRIVILEGED_DIM)},
            batch_size=[1],
        )
        # Construct the actual Stage 1 module and load every actor key, including its distribution.
        # Preserve the experiment seed for the independently initialized student.
        with torch.random.fork_rng(devices=[]):
            actor = PaperModel(
                obs, {"actor": ["policy", "privileged"]}, "actor", ACTION_DIM,
                hidden_dims=[128, 128], activation="lrelu", obs_normalization=True,
                distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"},
            )
        actor.load_state_dict(state_dict, strict=True)
        return cls(actor.to(device))

    def _normalize(self, raw: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return ((raw - self.mean[start:end]) * self.inv_std[start:end]).clamp(-10.0, 10.0)

    def normalized_history_pair(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One completed (x_t, a_t) pair, with the actor's fixed proprioceptive scaling."""
        return self._normalize(torch.cat((state, action), dim=-1), 0, PROPRIO_DIM)

    def latent(self, privileged: torch.Tensor) -> torch.Tensor:
        """Regression target from the current simulator-only environment measurements."""
        return self.encoder(self._normalize(privileged, PROPRIO_DIM, PROPRIO_DIM + PRIVILEGED_DIM))

    def action(self, policy: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Deterministic absolute joint-angle targets from (x_t, a_{t-1}) and an 8-D code."""
        return self.head(torch.cat((self._normalize(policy, 0, PROPRIO_DIM), latent), dim=-1))


class AdaptationModule(nn.Module):
    """Infer an 8-D environmental code from 50 completed state/action pairs, oldest first."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 128), nn.LeakyReLU(), nn.Linear(128, 32), nn.LeakyReLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1), nn.LeakyReLU(),
        )
        self.projection = nn.Linear(32 * 3, LATENT_DIM)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] != HISTORY_LENGTH or history.shape[2] != PROPRIO_DIM:
            raise ValueError("Expected a [batch, 50, 42] state/action history ordered from oldest to newest.")
        # Conv1d expects (batch, channels, time). Move the real time axis, keeping each frame intact.
        features = self.embedding(history).transpose(1, 2)
        return self.projection(self.temporal(features).flatten(start_dim=1))
