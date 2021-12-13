import random
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

from tic_tac_toe_343.common.b_models_and_buffer import ActorCritic, ReplayBuffer, Transition
from tic_tac_toe_343.common.d_utils import AGENT_TYPE


class TTTAgentA2C:
    def __init__(self, name, env, gamma=0.99, learning_rate=0.00001, batch_size=32):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=batch_size)
        self.agent_type = AGENT_TYPE.A2C.value

        self.actor_critic_model = ActorCritic(n_features=12, n_actions=12)
        self.optimizer = optim.Adam(self.actor_critic_model.parameters(), lr=learning_rate)

        # init rewards
        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

        self.model = self.actor_critic_model

    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        unavailable_actions = list(set(self.env.ALL_ACTIONS) - set(available_actions))
        obs = state.data.flatten()
        action = None

        # TODO
        action_prob = self.actor_critic_model.pi(obs)

        for i in unavailable_actions:
            action_prob[i] = 0

        m = Categorical(probs=action_prob)

        if mode == "TRAIN":
            coin = np.random.random()
            # epsilon보다 작으면 액션을 랜덤하게 뽑는다.
            if coin < epsilon:
                action = m.sample()
            else:
                action = torch.argmax(m.probs)
        else:
            action = torch.argmax(m.probs)

        return action.item()

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0

        self.buffer.append(
            Transition(state.data.flatten(), action, next_state.data.flatten(), reward, done)
        )

        if len(self.buffer) < self.batch_size:
            return loss

        # sample all from buffer
        batch = self.buffer.sample(batch_size=-1)

        observations, actions, next_observations, rewards, dones = batch

        # TODO
        self.optimizer.zero_grad()

        next_values = self.actor_critic_model.v(next_observations)
        td_target_value_lst = []

        for reward, next_value, done in zip(rewards, next_values, dones):
            td_target = reward + self.gamma * next_value * (0.0 if done else 1.0)
            td_target_value_lst.append(td_target)

        td_target_values = torch.tensor(td_target_value_lst, dtype=torch.float32).unsqueeze(dim=-1)

        values = self.actor_critic_model.v(observations)

        critic_loss = F.mse_loss(td_target_values.detach(), values)

        q_values = td_target_values
        advantages = (q_values - values).detach()

        action_probs = self.actor_critic_model.pi(observations)
        action_prob_selected = action_probs.gather(dim=1, index=actions)

        log_pi_advantages = torch.multiply(torch.log(action_prob_selected), advantages)

        log_actor_objective = torch.sum(log_pi_advantages)

        actor_loss = torch.multiply(log_actor_objective, -1.0)

        loss = critic_loss * 0.5 + actor_loss

        loss.backward()
        self.optimizer.step()
        self.training_time_steps += 1
        self.buffer.clear()

        return loss.item()
