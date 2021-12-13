import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import collections
from typing import Tuple

Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done']
)


class QNet(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(QNet, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        out = self.fc(x)
        return out


class Policy(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class ActorCritic(nn.Module):
    def __init__(self, n_features=4, n_actions=2, device=torch.device("cpu")):
        super(ActorCritic, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.fc_pi = nn.Linear(128, n_actions)
        self.fc_v = nn.Linear(128, 1)
        self.device = device

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        out = self.fc(x)
        return out

    def pi(self, x):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v


class ReplayBuffer:
    def __init__(self, capacity, device=torch.device("cpu")):
        self.buffer = collections.deque(maxlen=capacity)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> Tuple:
        if batch_size == -1:
            # Sample
            observations, actions, next_observations, rewards, dones = zip(*self.buffer)
        else:
            # Get index
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

            # Sample
            observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (64, 4), (64, 4)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards

        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (64, 1) (64, 1) (64,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return observations, actions, next_observations, rewards, dones
