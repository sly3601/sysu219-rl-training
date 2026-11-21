"""RSL-RL configuration for the 2021 RMA base policy and environment encoder."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SYSU219RMAPaperRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Each update uses 4,096 environments × 20 transitions; episodes span updates."""

    experiment_name = "sysu219_rma_paper_stage1"
    num_steps_per_env = 20
    max_iterations = 15_000
    save_interval = 50
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged"]}
    actor = RslRlMLPModelCfg(
        class_name=f"{__package__}.paper_ppo:PaperModel",
        hidden_dims=[128, 128],
        activation="lrelu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        class_name=f"{__package__}.paper_ppo:PaperModel",
        hidden_dims=[128, 128],
        activation="lrelu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name=f"{__package__}.paper_ppo:PaperPPO",
        clip_param=0.2,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        optimizer="adam",
        schedule="fixed",
        gamma=0.998,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=None,
        max_grad_norm=0.5,
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
    )


@configclass
class SYSU219RMAStage1V1RunnerCfg(SYSU219RMAPaperRunnerCfg):
    experiment_name = "sysu219_rma_stage1_v1"


@configclass
class SYSU219RMAStage1V2RunnerCfg(SYSU219RMAStage1V1RunnerCfg):
    experiment_name = "sysu219_rma_stage1_v2"
