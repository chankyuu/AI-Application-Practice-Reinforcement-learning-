import random
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F

from tic_tac_toe_343.common.b_models_and_buffer import QNet, ReplayBuffer, Transition
from tic_tac_toe_343.common.d_utils import AGENT_TYPE


class TTTAgentDqn:
    def __init__(
            self, name, env, gamma=0.99, learning_rate=0.001,
            replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
            min_buffer_size_for_training=100
    ):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_sync_step_interval = target_sync_step_interval
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size_for_training = min_buffer_size_for_training
        self.agent_type = AGENT_TYPE.DQN.value

        # network
        self.q = QNet(n_features=12, n_actions=12)
        self.target_q = QNet(n_features=12, n_actions=12)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

        self.model = self.q

    # 선택할 수 있는 액션 뽑아오는 함수
    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        unavailable_actions = list(set(self.env.ALL_ACTIONS) - set(available_actions))

        # 환경에서 state 객체가 존재함(내부적으로 data를 가지고있다(2차원 배열로 3 by 4)) <우리가 생각하는 화면>
        # flatten() -> 넘파이 함수인데, obs는 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 이런식으로 바꿔주는거임
        obs = state.data.flatten()
        action = None

        # TODO
        out = self.q.forward(obs)

        # 불가능한 actions는 모두 유한하지 않은 값으로 변경
        for i in unavailable_actions:
            out[i] = -np.inf

        # 학습 모드일때
        if mode == "TRAIN":
            coin = np.random.random()
            # epsilon보다 작으면 액션을 랜덤하게 뽑는다.
            if coin < epsilon:
                return np.random.randint(low=0, high=12)
            # 그렇지 않으면 argmax를 사용해서 가장 큰 값에 대응되는 인덱스 선택
            else:
                action = out.argmax(dim=0)
        # 학습 모드가 아닐때에는 항상 가장 큰 값에 대응되는 인덱스 선택
        else:
            action = out.argmax(dim=0)

        return action.numpy()

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0
        self.replay_buffer.append(
            Transition(state.data.flatten(), action, next_state.data.flatten(), reward, done)
        )
        if len(self.replay_buffer) < self.min_buffer_size_for_training:
            return loss

        batch = self.replay_buffer.sample(self.batch_size)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch

        # TODO
        # 타겟값과 현재 상태의 값의 오차를 최소화 시키기 위한 코드(mse)
        current_action_values = self.q(observations).gather(dim=1, index=actions)

        with torch.no_grad():
            next_action_values = self.target_q(next_observations).max(dim=1, keepdim=True).values
            next_action_values[dones] = 0.0
            next_action_values = next_action_values.detach()

            target_action_values = rewards + self.gamma * next_action_values

        loss = F.mse_loss(target_action_values, current_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_time_steps += 1

        # 타겟값 업데이트
        if self.training_time_steps % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()

