import argparse
import logging
import os

from gail.model_args import ModelArguments
from gail.utils import parse_config, setup_logging, set_wandb, set_all_seeds

logger = logging.getLogger(__name__)


class GailExecutor:
    def __init__(self, args):
        self.args = args
        os.environ["WANDB_MODE"] = self.args.wandb_mode
        set_wandb(self.args.wandb_dir)
        logger.info("args: {0}".format(self.args.to_json_string()))
        set_all_seeds(self.args.seed)

    def run(self):
        print("printing: {}".format(self.args))


def main(args):
    setup_logging()
    model_args = parse_config(ModelArguments, args.config)
    executor = GailExecutor(model_args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Model Runner")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
