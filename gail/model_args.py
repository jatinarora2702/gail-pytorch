import json
import os
from enum import Enum
from typing import Optional

import dataclasses
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name: str = field(default="gail", metadata={"help": "model identifier"})
    train: bool = field(default=True, metadata={"help": "do training. If set to False, we load existing model"})
    resume: Optional[str] = field(default=None, metadata={"help": "checkpoint to resume. Starts from scratch, if None"})
    discount_factor: float = field(default=0.99, metadata={"help": "discount factor"})
    clip_eps: float = field(default=0.2, metadata={"help": "clipping epsilon in PPO loss"})
    lr_actor: float = field(default=0.0003, metadata={"help": "actor model learning rate"})
    lr_critic: float = field(default=0.001, metadata={"help": "critic model learning rate"})
    lr_discriminator: float = field(default=0.001, metadata={"help": "discriminator model learning rate"})
    train_steps: int = field(default=1e5, metadata={"help": "maximum training time steps"})
    max_episode_len: int = field(default=400, metadata={"help": "maximum episode length"})
    update_steps: int = field(default=1600, metadata={"help": "frequency of model update"})
    checkpoint_steps: int = field(default=2e4, metadata={"help": "frequency of model saving"})
    num_epochs: int = field(default=40, metadata={"help": "training epochs of PPO model"})
    num_d_epochs: int = field(default=2, metadata={"help": "training epochs of discriminator model"})

    out_root: str = field(default="../out", metadata={"help": "outputs root directory"})
    expert_trajectory_dir: str = field(default="../trajectory", metadata={"help": "expert tranjectory dir"})
    env_id: str = field(default="cartpole", metadata={"help": "simulation environment identifier"})
    wandb_mode: str = field(default="run", metadata={"help": "can enable/disable wandb online sync (run/dryrun)"})
    seed: int = field(default=42, metadata={"help": "random seed for reproducibility of results"})
    device: str = field(default="cuda:0", metadata={"help": "device (cpu|cuda:0)"})

    def __post_init__(self):
        self.run_root = os.path.join(self.out_root, self.env_id, self.model_name)
        if self.resume:
            self.resume = os.path.join(self.run_root, "checkpoints", "checkpoint-{0}".format(self.resume))
        self.wandb_dir = self.run_root

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support).
        """
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)
