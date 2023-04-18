import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from typing import NamedTuple, List
from torch.optim.lr_scheduler import _LRScheduler
import math

torch.manual_seed(12321)
np.random.seed(12321)


class Agent:
    """
    A reinforcement learning agent that learns to play a Lunar Lander game.
    https://gymnasium.farama.org/environments/box2d/lunar_lander/

    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        lr: float = 0.0018,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        tau: float = 1e-2,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = 64
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.n_action = self.action_space.n
        self.n_observ = self.observation_space.shape[0]
        self.q_net = QNet(self.n_observ, self.n_action)
        self.target_net = QNet(self.n_observ, self.n_action)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.lr_scheduler = CustomLRScheduler(self.optimizer)
        self.loss_fn = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.n_observ, self.n_action, 1000000)
        self.tau = tau
        self.current_action = []
        self.current_observation = []

        for p in self.target_net.parameters():
            p.requires_grad = False

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Take an observation and return an action.

        Args:
            observation (gym.spaces.Box): _description_

        Returns:
            gym.spaces.Discrete: _description_
        """
        self.current_observation = observation
        # return self.action_space.sample()
        if np.random.uniform() < self.epsilon:
            # Randomly select an action with probability exploration_rate.
            self.current_action = self.action_space.sample()
        else:
            # Select the action with the highest Q-value for the given observation.
            state = torch.from_numpy(observation)
            with torch.no_grad():
                q_values = self.q_net(state)
            self.current_action = torch.argmax(q_values)
        return self.current_action.item()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
         Take an observation, a reward, a boolean indicating whether the episode has
         terminated, and a boolean indicating whether the episode was truncated

        Args:
            observation (gym.spaces.Box): _description_
            reward (float): _description_
            terminated (bool): _description_
            truncated (bool): _description_
        """
        # pass
        self.replay_buffer.add(
            self.current_observation,
            self.current_action,
            reward,
            observation,
            terminated,
            truncated,
        )
        # # Don't train if we aren't ready
        # if frame < self.train_start:
        #     return
        # elif frame == self.train_start:
        #     self.is_waiting = False

        if self.replay_buffer.n_samples < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        done = (torch.logical_or(batch.terminated, batch.truncated)).type(torch.float32)
        q_actions = self.q_net(batch.state)
        q_pred = q_actions.gather(1, batch.action)
        with torch.no_grad():
            q_target_actions = self.target_net(batch.next_state)
            q_target = q_target_actions.max(dim=1)[0].view(-1, 1)
            q_target = batch.reward + self.gamma * (1 - done) * q_target
        loss = self.loss_fn(q_target, q_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.soft_update_from_to()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update_from_to(self):
        for target_param, param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


class Batch(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim, act_dim, buffer_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.n_samples = 0

        self.state = torch.zeros(buffer_size, state_dim, dtype=torch.float32)
        self.action = torch.zeros(buffer_size, 1, dtype=torch.int64)
        self.reward = torch.zeros(buffer_size, 1, dtype=torch.float32)
        self.next_state = torch.zeros(buffer_size, state_dim, dtype=torch.float32)
        self.terminated = torch.zeros(buffer_size, 1, dtype=torch.float32)
        self.truncated = torch.zeros(buffer_size, 1, dtype=torch.float32)

    def add(self, state, action, reward, next_state, terminated, truncated):
        self.state[self.ptr] = torch.from_numpy(state)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = torch.from_numpy(next_state)
        self.terminated[self.ptr] = terminated
        self.truncated[self.ptr] = truncated

        if self.n_samples < self.buffer_size:
            self.n_samples += 1

        self.ptr = (self.ptr + 1) % self.buffer_size

    def sample(self, batch_size):
        # Select batch_size number of sample indicies at random from the buffer
        idx = np.random.choice(self.n_samples, batch_size)
        # Using the random indices, assign the corresponding state, action, reward,
        # discount, and next state samples.
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        next_state = self.next_state[idx]
        terminated = self.terminated[idx]
        truncated = self.truncated[idx]

        return Batch(state, action, reward, next_state, terminated, truncated)

    def last(self):
        # Select batch_size number of sample indicies at random from the buffer
        idx = self.n_samples - 1
        # Using the random indices, assign the corresponding state, action, reward,
        # discount, and next state samples.
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        next_state = self.next_state[idx]
        terminated = self.terminated[idx]
        truncated = self.truncated[idx]

        return Batch(state, action, reward, next_state, terminated, truncated)


class QNet(nn.Module):
    def __init__(self, n_observ, n_action):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observ, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_action),
        )
        # nn.init.xavier_uniform_(self.model[0].weight)
        # nn.init.xavier_uniform_(self.model[2].weight)
        # nn.init.xavier_uniform_(self.model[4].weight)

    def forward(self, x):
        return self.model(x)


class CustomLRScheduler(_LRScheduler):
    """
    A custom defined learning rate sheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: int = 0.95,
        T_0: int = 32,
        T_mult: int = 32,
        eta_min: int = 0.00023,
        last_epoch: int = -1,
    ) -> None:
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.
        """
        if gamma is not None:
            self.gamma = gamma
        if T_0 is not None:
            self.T_0 = T_0
            self.T_i = T_0
        if T_mult is not None:
            self.T_mult = T_mult
        if eta_min is not None:
            self.eta_min = eta_min

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """_summary_
        Returns:
            List[float]: _description_
        """
        self.T_i *= self.T_mult ** (self.last_epoch // self.T_i)
        return [
            self.eta_min
            + (base_lr * self.gamma ** (self.last_epoch // self.T_i) - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch % self.T_i) / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
