import argparse
import logging
import os

import gym
import numpy as np
import torch

from gail.model_args import ModelArguments
from gail.ppo import PolicyModel
from gail.utils import parse_config, setup_logging, set_wandb, set_all_seeds

logger = logging.getLogger(__name__)


class PpoExecutor:
    def __init__(self, args):
        self.args = args

        os.environ["WANDB_MODE"] = self.args.wandb_mode
        set_wandb(self.args.wandb_dir)
        logger.info("args: {0}".format(self.args.to_json_string()))
        set_all_seeds(self.args.seed)

        self.env = gym.make(self.args.env_id)
        self.env.seed(self.args.seed)
        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.num_actions = self.env.action_space.n

        self.policy = PolicyModel(self.args).to(self.args.device)
        self.policy_old = PolicyModel(self.args).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.states = []
        self.actions = []

    def reset_buffers(self):
        self.states = []
        self.actions = []

    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.args.device)
        with torch.no_grad():
            action, action_log_prob = self.policy_old.act(state_tensor)
        action = action.detach().item()

        self.states.append(state)
        self.actions.append(action)
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done

    def run(self):
        for t in range(self.args.num_trajectories):
            state = self.env.reset()
            for ep in range(self.args.max_episode_len):
                state, reward, done = self.take_action(state)
                if done:
                    break

            PpoExecutor.save_to_file(self.states, "{0}/trajectory/states.csv".format(self.args.env_root))
            PpoExecutor.save_to_file(self.actions, "{0}/trajectory/actions.csv".format(self.args.env_root))
            self.reset_buffers()

    def load(self, checkpoint_dir):
        policy_model_path = "{0}/policy.ckpt".format(checkpoint_dir)
        self.policy_old.load_state_dict(torch.load(policy_model_path, map_location=lambda x, y: x))
        self.policy.load_state_dict(self.policy_old.state_dict())

    @staticmethod
    def save_to_file(data, file_path):
        try:
            with open(file_path, "ab") as handle:
                np.savetxt(handle, data, fmt="%s")
        except FileNotFoundError:
            with open(file_path, "wb") as handle:
                np.savetxt(handle, data, fmt="%s")


def main(args):
    setup_logging()
    model_args = parse_config(ModelArguments, args.config)
    executor = PpoExecutor(model_args)
    executor.load(executor.args.checkpoint_dir)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="sample trajectories from pretrained PPO model")
    ap.add_argument("--config", default="config/CartPole-v0/config_traj.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
