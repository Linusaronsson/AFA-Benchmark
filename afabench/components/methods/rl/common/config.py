from dataclasses import dataclass


@dataclass
class AFARLTrainingLoopConfig:
    frames_per_batch: int
    n_batches: int
    eval_max_steps: int
    n_eval_episodes: int
    device: str | None = None
    env_seed: int | None = None
    eval_n_times: int = 10


@dataclass
class AFAMDPConfig:
    hard_budget: int | None = None
    force_hard_budget: bool = True
    n_agents: int = 1
