import time

import numpy as np
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import multiprocessing as mp

from a_common.a_commons import VectorizedTransitions, make_env
from a_common.b_models import Policy
from a_common.c_buffers import ReplayBufferForVectorizedEnvs


def rl_main():
    n_envs = 4
    #num_cores = mp.cpu_count() # 본인 cpu 코어 수에 따라 env 설정
    #n_envs = num_cores - 1

    env = AsyncVectorEnv(env_fns=[make_env for _ in range(n_envs)])
    policy = Policy(n_features=4, n_actions=3)

    replay_buffer_for_vectorized_envs = ReplayBufferForVectorizedEnvs(
        capacity=10_000
    )

    total_time_steps = 10
    episode_rewards = np.zeros((n_envs,)) # [0, 0, 0, ....., 0]
    episode_reward_lst = []
    num_train_steps = 0
    total_train_start_time = time.time()

    observations = env.reset() # env 개수에 따라 한꺼번에 observations가 튀어 나옴

    for time_step in range(total_time_steps):
        actions = policy.get_action(observations) # env 개수에 따라 각각에 대해 action이 한번에 튀어나옴
        next_observations, rewards, dones, infos = env.step(actions)

        vectorized_transitions = VectorizedTransitions(
            observations, actions, next_observations, rewards, dones
        )
        replay_buffer_for_vectorized_envs.append(vectorized_transitions)

        # ex) episode_rewards에 이런식으로 들어가짐(numpy의 +=)
        # [0, 0, 0, 0] += [-1.0, -1.0, 10.0, -1.0]
        episode_rewards += rewards

        if len(replay_buffer_for_vectorized_envs) > 1:
            # TRAIN POLICY
            num_train_steps += 1

        print("[{0:>3}] Observations: {1}, Actions: {2}, Next Observations: {3}, "
              "Rewards: {4}, Dones: {5} || "
              "Replay Buffer: {6}, Training Steps: {7}".format(
            time_step + 1,
            str(np.array(observations).argmax(axis=1)),
            actions,
            str(np.array(next_observations).argmax(axis=1)),
            rewards,
            str(dones),
            replay_buffer_for_vectorized_envs.size(), num_train_steps
        ))

        if any(dones):
            # print(episode_rewards[dones], len(episode_rewards[dones]), "****")
            episode_reward_lst.extend([episode_reward for episode_reward in episode_rewards[dones]])
            episode_rewards[dones] = 0.0
        # print(episode_reward_lst, "##########################")

        observations = next_observations

    total_training_time = time.time() - total_train_start_time
    formatted_total_training_time = time.strftime(
        '%H:%M:%S', time.gmtime(total_training_time)
    )
    print("Total Training End : {}".format(formatted_total_training_time))
    print("Rate of Buffer Increase: {0:.3f}/1sec.".format(
        replay_buffer_for_vectorized_envs.size() / total_training_time
    ))
    print("Rate of Training Steps: {0:.3f}/1sec.".format(
        num_train_steps / total_training_time
    ))


if __name__ == "__main__":
    rl_main()
