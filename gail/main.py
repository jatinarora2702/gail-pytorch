import argparse
import logging
import os

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from gail.model_args import ModelArguments
from gail.utils import parse_config, setup_logging, set_wandb, set_all_seeds

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    def __init__(self, args):
        super(PolicyModel, self).__init__()
        self.args = args
        self.actor = nn.Sequential(
            nn.Linear(self.args.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.args.num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.args.state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob

    def evaluate(self, state, action):
        action_prob = self.actor(state)
        action_dist = Categorical(action_prob)
        action_log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        value = self.critic(state)
        return value, action_log_prob, entropy

    def forward(self):
        raise NotImplementedError


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(self.args.state_dim + self.args.num_actions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, state_action):
        reward = self.model(state_action)
        return reward


class GailExecutor:
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
        self.discriminator = Discriminator(self.args).to(self.args.device)

        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": self.args.lr_actor},
            {"params": self.policy.critic.parameters(), "lr": self.args.lr_critic}
        ])
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_discriminator)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        expert_states = np.genfromtxt("{0}/trajectory/states.csv".format(self.args.env_root))
        expert_states = torch.tensor(expert_states, dtype=torch.float32, device=self.args.device)
        expert_actions = np.genfromtxt("{0}/trajectory/actions.csv".format(self.args.env_root), dtype=np.int32)
        expert_actions = torch.tensor(expert_actions, dtype=torch.int64, device=self.args.device)
        expert_actions = torch.eye(self.args.num_actions)[expert_actions].to(self.args.device)
        self.expert_state_actions = torch.cat([expert_states, expert_actions], dim=1)

        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
        self.is_terminal = []

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.log_prob_actions = []
        self.rewards = []
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
        prev_states = torch.stack(self.states, dim=0).to(self.args.device)
        prev_actions = torch.stack(self.actions, dim=0).to(self.args.device)
        prev_log_prob_actions = torch.stack(self.log_prob_actions, dim=0).to(self.args.device)
        prev_actions_one_hot = torch.eye(self.args.num_actions)[prev_actions.long()].to(self.args.device)
        agent_state_actions = torch.cat([prev_states, prev_actions_one_hot], dim=1)

        for ep in range(self.args.num_d_epochs):
            expert_prob = self.discriminator(self.expert_state_actions)
            agent_prob = self.discriminator(agent_state_actions)
            term1 = self.bce_loss(agent_prob, torch.ones((agent_state_actions.shape[0], 1), device=self.args.device))
            term2 = self.bce_loss(expert_prob, torch.zeros((self.expert_state_actions.shape[0], 1),
                                                           device=self.args.device))
            loss = term1 + term2
            self.d_optimizer.zero_grad()
            loss.backward()
            self.d_optimizer.step()

        with torch.no_grad():
            d_rewards = torch.log(self.discriminator(agent_state_actions))

        rewards = []
        cumulative_discounted_reward = 0.
        for i in range(len(d_rewards) - 1, -1, -1):
            cumulative_discounted_reward = d_rewards[i] + self.args.discount_factor * cumulative_discounted_reward
            rewards.append(cumulative_discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.args.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        for ep in range(self.args.num_epochs):
            values, log_prob_actions, entropy = self.policy.evaluate(prev_states, prev_actions)
            advantages = rewards - values.detach()
            imp_ratios = torch.exp(log_prob_actions - prev_log_prob_actions)
            clamped_imp_ratio = torch.clamp(imp_ratios, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
            term1 = -torch.min(imp_ratios, clamped_imp_ratio) * advantages
            term2 = 0.5 * self.mse_loss(values, rewards)
            term3 = -0.01 * entropy
            loss = term1 + term2 + term3
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.reset_buffers()

    def run(self):
        t = 1
        success_count = 0
        finish = False
        while t <= self.args.train_steps:
            state = self.env.reset()
            total_reward = 0
            done = False
            for ep in range(self.args.max_episode_len):
                state, reward, done = self.take_action(state)
                total_reward += reward
                if self.args.train and t % self.args.update_steps == 0:
                    logger.info("updating model")
                    self.update()
                if self.args.train and t % self.args.checkpoint_steps == 0:
                    logger.info("saving checkpoint")
                    self.save(self.args.checkpoint_dir)
                t += 1
                if done:
                    if total_reward >= self.args.reward_threshold:
                        success_count += 1
                        if success_count >= 100:
                            logger.info("model trained. saving checkpoint")
                            self.save(self.args.checkpoint_dir)
                            finish = True
                    else:
                        success_count = 0
                    logger.info("iter: {0} | reward: {1:.1f}".format(t, total_reward))
                    if not self.args.train:
                        self.reset_buffers()
                    break
            if not done:
                logger.info("truncated at horizon")
            if finish:
                break

    def save(self, checkpoint_dir):
        torch.save(self.policy_old.state_dict(), "{0}/policy.ckpt".format(checkpoint_dir))
        torch.save(self.discriminator.state_dict(), "{0}/discriminator.ckpt".format(checkpoint_dir))

    def load(self, checkpoint_dir):
        policy_model_path = "{0}/policy.ckpt".format(checkpoint_dir)
        self.policy_old.load_state_dict(torch.load(policy_model_path, map_location=lambda x, y: x))
        self.policy.load_state_dict(self.policy_old.state_dict())
        discriminator_model_path = "{0}/discriminator.ckpt".format(checkpoint_dir)
        self.discriminator.load_state_dict(torch.load(discriminator_model_path, map_location=lambda x, y: x))


def main(args):
    setup_logging()
    model_args = parse_config(ModelArguments, args.config)
    executor = GailExecutor(model_args)
    if not executor.args.train:
        executor.load(executor.args.checkpoint_dir)
    executor.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GAIL model")
    ap.add_argument("--config", default="config/LunarLander-v2/config_gail.json", help="config json file")
    ap = ap.parse_args()
    main(ap)
