import json
import logging
import os
import random
from pathlib import Path

import dataclasses
import numpy as np
import torch
import wandb


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)


def set_wandb(wandb_dir):
    os.environ["WANDB_WATCH"] = "all"
    os.makedirs(os.path.join(wandb_dir, "wandb"), exist_ok=True)
    wandb.init(project=os.getenv("WANDB_PROJECT", "gail-pytorch"), dir=wandb_dir)


def parse_config(args_class, json_file):
    data = json.loads(Path(json_file).read_text())

    # curr_run_output_dir = os.path.join(data["out_root"], data["dataset_dir"], data["model_name"])
    # data["output_dir"] = os.path.join(curr_run_output_dir, "checkpoints")
    # data["logging_dir"] = os.path.join(curr_run_output_dir, default_logdir())

    keys = {f.name for f in dataclasses.fields(args_class)}
    inputs = {k: v for k, v in data.items() if k in keys}
    return args_class(**inputs)
