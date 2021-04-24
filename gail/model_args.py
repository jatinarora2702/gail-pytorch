import json
import os
from enum import Enum
from typing import Optional

import dataclasses
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name: str = field(default="gail", metadata={"help": "model identifier"})
    resume: Optional[str] = field(default=None, metadata={"help": "checkpoint to resume. Starts from scratch, if None"})
    out_root: str = field(default="../out", metadata={"help": "outputs root directory"})
    env_id: str = field(default="cartpole", metadata={"help": "simulation environment identifier"})
    wandb_mode: str = field(default="run", metadata={"help": "can enable/disable wandb online sync (run/dryrun)"})
    seed: int = field(default=42, metadata={"help": "random seed for reproducibility of results"})

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
