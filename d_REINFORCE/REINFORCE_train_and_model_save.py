import sys
import os
import time

import gym
import numpy as np
import torch
import torch.optim as optim

import wandb

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from a_common.b_models import Policy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_DIR = os.path.join(PROJECT_HOME, "d_REINFORCE", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class REINFORCE:
    def __init__(
            self, env_name, env, test_env, use_wandb, wandb_entity,
            max_num_episodes, learning_rate, gamma,
            print_episode_interval,
            test_episode_interval, test_num_episodes,
            episode_reward_avg_solved, episode_reward_std_solved
    ):
        self.env_name = env_name
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb = wandb.init(
                entity=wandb_entity,
                project="REINFOCE_{0}".format(self.env_name)
            )
        self.max_num_episodes = max_num_episodes
        self.gamma = gamma
        self.print_episode_interval = print_episode_interval
        self.test_episode_interval = test_episode_interval
        self.test_num_episodes = test_num_episodes
        self.episode_reward_avg_solved = episode_reward_avg_solved
        self.episode_reward_std_solved = episode_reward_std_solved

        # env
        self.env = env
        self.test_env = test_env

        self.buffer = []

        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # init rewards
        self.episode_reward_lst = []
        self.best_test_mean_episode_reward = -1000000000

        self.time_steps = 0
        self.training_time_steps = 0

    def train_loop(self):
        total_train_start_time = time.time()

        test_episode_reward_avg = 0.0
        test_episode_reward_std = 0.0

        is_terminated = False

        for n_episode in range(self.max_num_episodes):
            episode_start_time = time.time()
            episode_reward = 0

            # Environment ???????????? ?????? ?????????
            observation = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1
                action, action_prob_selected = self.policy.get_action_with_action_prob(
                    observation
                )

                next_observation, reward, done, _ = self.env.step(action)

                self.buffer.append((reward, action_prob_selected))
                observation = next_observation
                episode_reward += reward

            # TRAIN
            objective = self.train_step()

            self.episode_reward_lst.append(episode_reward)

            mean_episode_reward = np.mean(self.episode_reward_lst[-100:])

            if self.training_time_steps > 0 and n_episode % self.test_episode_interval == 0:
                test_episode_reward_avg, test_episode_reward_std = self.reinforce_testing(
                    self.test_num_episodes
                )

                print("[Test Episode Reward] Average: {0:.3f}, Standard Dev.: {1:.3f}".format(
                    test_episode_reward_avg, test_episode_reward_std
                ))

                termination_conditions = [
                    test_episode_reward_avg > self.episode_reward_avg_solved,
                    test_episode_reward_std < self.episode_reward_std_solved
                ]

                if all(termination_conditions):
                    print("Solved in {0} steps ({1} training steps)!".format(
                        self.time_steps, self.training_time_steps
                    ))
                    self.model_save(
                        test_episode_reward_avg, test_episode_reward_std
                    )
                    is_terminated = True

            per_episode_time = time.time() - episode_start_time
            per_episode_time = time.strftime('%H:%M:%S', time.gmtime(per_episode_time))

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime(
                '%H:%M:%S', time.gmtime(total_training_time)
            )

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3}, Steps {:6}, Training Steps {:6}]".format(
                        n_episode + 1, self.time_steps, self.training_time_steps
                    ),
                    "Episode Reward: {:>5},".format(episode_reward),
                    "Mean Episode Reward: {:.3f},".format(mean_episode_reward),
                    "Objective: {:.3f},".format(objective),
                    "Per-Episode Time: {}".format(per_episode_time),
                    "Total Elapsed Time {}".format(total_training_time)
                )

            if self.use_wandb:
                self.wandb.log({
                    "[TEST] Average Episode Reward": test_episode_reward_avg,
                    "[TEST] Std. Episode Reward": test_episode_reward_std,
                    "Episode Reward": episode_reward,
                    "Episode": n_episode,
                    "Objective": objective if objective != 0.0 else 0.0,
                    "Mean Episode Reward": mean_episode_reward,
                    "Number of Training Steps": self.training_time_steps,
                })

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

    def train_step(self):
        self.training_time_steps += 1

        G = 0
        policy_gradient = torch.tensor(0.0, dtype=torch.float32)

        self.optimizer.zero_grad()

        for r, prob in self.buffer[::-1]:
            G = r + self.gamma * G
            policy_gradient += torch.log(prob)

        policy_gradient = torch.multiply(policy_gradient, -1.0 * G)

        policy_gradient.backward()
        self.optimizer.step()
        self.buffer = []

        return G

    def train_step_2(self):
        self.training_time_steps += 1

        G = 0
        policy_gradient = torch.tensor(0.0, dtype=torch.float32)

        self.optimizer.zero_grad()

        for r, prob in self.buffer[::-1]:
            G = r + self.gamma * G
            policy_gradient += torch.log(prob) * G

        policy_gradient = torch.multiply(policy_gradient, -1.0)

        policy_gradient.backward()
        self.optimizer.step()
        self.buffer = []

        return G

    def model_save(self, test_episode_reward_avg, test_episode_reward_std):
        print("Solved in {0} steps ({1} training steps)!".format(
            self.time_steps, self.training_time_steps
        ))
        torch.save(
            self.policy.state_dict(),
            os.path.join(MODEL_DIR, "reinforce_{0}_{1:4.1f}_{2:3.1f}.pth".format(
                self.env_name, test_episode_reward_avg, test_episode_reward_std
            ))
        )

    def reinforce_testing(self, num_episodes):
        episode_reward_lst = []

        for i in range(num_episodes):
            episode_reward = 0  # cumulative_reward

            # Environment ???????????? ?????? ?????????
            observation = self.test_env.reset()

            while True:
                action = self.policy.get_action(observation, mode="test")

                next_observation, reward, done, _ = self.test_env.step(action)

                episode_reward += reward  # episode_reward ??? ???????????? ????????? ????????? ???????????? ?????? ??? ????????? ??? ?????????.
                observation = next_observation

                if done:
                    break

            episode_reward_lst.append(episode_reward)

        return np.average(episode_reward_lst), np.std(episode_reward_lst)


def main():
    ENV_NAME = "CartPole-v1"

    # env
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    reinforce = REINFORCE(
        env_name=ENV_NAME,
        env=env,
        test_env=test_env,
        use_wandb=False,                            # WANDB ?????? ??? ?????? ??????
        wandb_entity="chankyuu",          # WANDB ?????? ??????
        max_num_episodes=10_000,                 # ????????? ?????? ?????? ???????????? ??????
        learning_rate=0.0002,                   # ?????????
        gamma=0.99,                             # ?????????
        print_episode_interval=10,               # Episode ?????? ????????? ?????? ???????????? ??????
        test_episode_interval=50,                # ???????????? ?????? episode ??????
        test_num_episodes=3,                    # ??????????????? ???????????? ???????????? ??????
        episode_reward_avg_solved=450,          # ?????? ????????? ?????? ????????? ???????????? ???????????? Average
        episode_reward_std_solved=10            # ?????? ????????? ?????? ????????? ???????????? ???????????? Standard Deviation
    )
    reinforce.train_loop()


if __name__ == '__main__':
    main()