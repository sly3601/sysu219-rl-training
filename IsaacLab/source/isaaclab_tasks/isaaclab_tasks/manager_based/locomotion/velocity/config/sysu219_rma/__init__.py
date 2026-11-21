"""RMA phase-one expert for the SYSU219 robot."""

import gymnasium as gym

gym.register(
    id="Isaac-RMA-SYSU219-Stage1-v0",
    entry_point=f"{__name__}.paper_env:PaperRMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_env:SYSU219RMAPaperEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.paper_agent:SYSU219RMAPaperRunnerCfg",
    },
)

gym.register(
    id="Isaac-RMA-SYSU219-Stage1-v1",
    entry_point=f"{__name__}.paper_env:PaperRMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_env:SYSU219RMAStage1V1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.paper_agent:SYSU219RMAStage1V1RunnerCfg",
    },
)

gym.register(
    id="Isaac-RMA-SYSU219-Stage1-v2",
    entry_point=f"{__name__}.paper_env:PaperRMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.paper_env:SYSU219RMAStage1V2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.paper_agent:SYSU219RMAStage1V2RunnerCfg",
    },
)
