import argparse
import logging
import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gail.model_args import ModelArguments
from gail.utils import parse_config, setup_logging, set_wandb, set_all_seeds

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    def __init__(self, args):
        super(PolicyModel, self).__init__()
        self.args = args
        self.actor = nn.Linear(self.args.state_dim, self.args.num_actions)
        self.critic = nn.Linear(self.args.state_dim, 1)  # check: whether 1 or 2(num_actions)?

    def act(self, state):
        x = self.actor(state)
        action_prob = F.softmax(x, dim=-1)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob

    def evaluate(self, state, action):
        x = self.actor(state)
        action_prob = F.softmax(x, dim=-1)
        action_dist = Categorical(action_prob)
        action_log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        x = self.critic(state)
        value = F.tanh(x)

        return value, action_log_prob, entropy

    def forward(self):
        raise NotImplementedError


class PpoExecutor:
    def __init__(self, args):
        self.args = args
        self.discount_factor = 0.99
        self.clip_eps = 0.2
        self.lr_actor = 0.0003
        self.lr_critic = 0.001
        self.train_steps = int(1e5)
        self.max_episode_len = 400
        self.update_steps = self.max_episode_len * 4
        self.num_epochs = 40
        self.checkpoint_steps = int(2e4)

        os.environ["WANDB_MODE"] = self.args.wandb_mode
        set_wandb(self.args.wandb_dir)
        logger.info("args: {0}".format(self.args.to_json_string()))
        set_all_seeds(self.args.seed)

        self.env = gym.make("CartPole-v0")
        self.env.seed(self.args.seed)
        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.num_actions = self.env.action_space.n

        self.policy = PolicyModel(self.args).to(self.args.device)
        self.policy_old = PolicyModel(self.args).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": self.lr_actor},
            {"params": self.policy.critic.parameters(), "lr": self.lr_critic}
        ])
        self.mse_loss = nn.MSELoss()

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.values = []
        self.is_terminal = []

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.args.device)
        with torch.no_grad():
            action, action_log_prob = self.policy_old.act(state)
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_prob_actions.append(action_log_prob.detach())

        action = action.detach().item()
        next_state, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.is_terminal.append(done)

        return next_state, reward, done

    def update(self):
        rewards = []
        cumulative_discounted_reward = 0.
        for i in range(len(self.rewards) - 1, 0, -1):
            if self.is_terminal[i]:
                cumulative_discounted_reward = 0.
            cumulative_discounted_reward = self.rewards[i] + self.discount_factor * cumulative_discounted_reward
            rewards.append(cumulative_discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float64, device=self.args.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        prev_states = torch.stack(self.states, dim=0).to(self.args.device)
        prev_actions = torch.stack(self.actions, dim=0).to(self.args.device)
        prev_log_prob_actions = torch.stack(self.log_prob_actions, dim=0).to(self.args.device)

        for ep in range(self.num_epochs):
            values, log_prob_actions, entropy = self.policy.evaluate(prev_states, prev_actions)
            advantages = rewards - values.detach()
            imp_ratios = torch.exp(log_prob_actions - prev_log_prob_actions)
            term1 = -torch.min(imp_ratios, torch.clamp(imp_ratios, 1 - self.clip_eps, 1 + self.clip_eps)) * advantages
            term2 = 0.5 * self.mse_loss(values, rewards)
            term3 = -0.01 * entropy
            loss = term1 + term2 + term3
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

    def run(self):
        t = 1
        while t <= self.train_steps:
            state = self.env.reset()
            total_reward = 0
            for ep in range(self.max_episode_len):
                state, reward, done = self.take_action(state)
                total_reward += reward
                if t % self.update_steps == 0:
                    self.update()
                if t % self.checkpoint_steps == 0:
                    self.save("../out/cartpole")
                if done:
                    break
            logger.info("total reward: {0:.4f}".format(total_reward))

    def save(self, checkpoint_location):
        torch.save(self.policy_old.state_dict(), checkpoint_location)

    def load(self, checkpoint_location):
        self.policy_old.load_save_dict(torch.load(checkpoint_location, map_location=lambda x, y: x))
        self.policy.load_save_dict(self.policy_old.state_dict())


def main(args):
    setup_logging()
    model_args = parse_config(ModelArguments, args.config)
    executor = PpoExecutor(model_args)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PPO for sampling expert trajectories")
    ap.add_argument("--config", default="config/config_debug.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
