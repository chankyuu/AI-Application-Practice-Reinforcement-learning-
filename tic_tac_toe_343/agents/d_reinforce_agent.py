import random
import torch
from torch import optim
import numpy as np
from torch.distributions import Categorical

from tic_tac_toe_343.common.b_models_and_buffer import Policy, Transition, ReplayBuffer
from tic_tac_toe_343.common.d_utils import AGENT_TYPE


class TTTAgentReinforce:
    def __init__(self, name, env, gamma=0.99, learning_rate=0.001):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.agent_type = AGENT_TYPE.REINFORCE.value

        self.buffer = ReplayBuffer(capacity=100_000)

        self.policy = Policy(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

        self.model = self.policy

    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        unavailable_actions = list(set(self.env.ALL_ACTIONS) - set(available_actions))
        obs = state.data.flatten()
        action = None

        # TODO
        # 정책
        action_prob = self.policy.forward(obs)

        # 불가능한 액션 처리
        for i in unavailable_actions:
            action_prob[i] = 0.0

        m = Categorical(probs=action_prob)

        # 학습 모드일때
        if mode == "TRAIN":
            m = Categorical(probs=action_prob)
            action = m.sample()
        else :
            action = torch.argmax(action_prob, dim=-1)

        return action.item()

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0
        self.buffer.append(
            Transition(state.data.flatten(), action, next_state.data.flatten(), reward, done)
        )
        if not done or len(self.buffer) == 0:
            return loss

        # sample all from buffer
        batch = self.buffer.sample(batch_size=-1)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, _, rewards, _ = batch

        # TODO
        action_probs = self.policy.forward(observations)
        action_probs_selected = action_probs.gather(dim=-1, index=actions)
        action_probs_selected = action_probs_selected.squeeze(dim=-1)
        # 누적 보상 계산
        G = 0
        return_lst = []
        rewards = torch.flip(rewards, dims=(0,))

        for reward in rewards:
            G = reward + self.gamma * G
            return_lst.append(G)
        return_lst = torch.tensor(return_lst[::-1], dtype=torch.float32)

        log_pi_returns = torch.multiply(torch.log(action_probs_selected), return_lst)
        log_policy_objective = torch.sum(log_pi_returns)
        loss = torch.multiply(log_policy_objective, -1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_time_steps += 1
        self.buffer.clear()

        return loss.item()
