# 선수 에이전트: RL 에이전트, 후수 에이전트: Dummy 에이전트
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time

from tic_tac_toe_343.agents.a_dummy_agent import Dummy_Agent
from tic_tac_toe_343.common.a_env_tic_tac_toe_343 import TicTacToe343
from tic_tac_toe_343.common.c_game_stats import draw_performance, print_game_statistics, \
    print_step_status, epsilon_scheduled, GameStatus
from tic_tac_toe_343.common.d_utils import PLAY_TYPE

# from agents.c_dqn_agent import TTTAgentDqn
from tic_tac_toe_343.agents.c_dqn_agent import TTTAgentDqn

# from agents.d_reinforce_agent import TTTAgentReinforce
from tic_tac_toe_343.agents.d_reinforce_agent import TTTAgentReinforce

# from agents.e_a2c_agent import TTTAgentA2C
from tic_tac_toe_343.agents.e_a2c_agent import TTTAgentA2C


INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 20_000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 30_000

STEP_VERBOSE = False
BOARD_RENDER = False


# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Dummy 에이전트
def learning_for_agent_1_vs_agent_2():
    # Create environment
    game_status = GameStatus()
    env = TicTacToe343()

    ##################
    # Create agent_1 #
    ##################

    play_type = PLAY_TYPE.FIRST

    # Create agent
    agent_1 = Dummy_Agent(name="AGENT_1", env=env)

    # agent_1 = TTTAgentDqn(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001,
    #     replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=1_000,
    #     min_buffer_size_for_training=1_000
    # )

    # agent_1 = TTTAgentReinforce(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001
    # )

    # agent_1 = TTTAgentA2C(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001, batch_size=32
    # )

    # agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    agent_2 = TTTAgentDqn(
        name="AGENT_2", env=env, gamma=0.99, learning_rate=0.00001,
        replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=1_000,
        min_buffer_size_for_training=1_000
    )

    # agent_2 = TTTAgentReinforce(
    #     name="AGENT_2", env=env, gamma=0.99, learning_rate=0.00001
    # )

    # agent_2 = TTTAgentA2C(
    #     name="AGENT_2", env=env, gamma=0.99, learning_rate=0.00001, batch_size=32
    # )

    total_steps = 0

    # Episode iteration
    for episode in range(1, MAX_EPISODES + 1):
        curr_state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        prev_state = None
        prev_action = None

        agent_1_episode_td_error = 0.0
        agent_2_episode_td_error = 0.0

        curr_agent = agent_1
        oppo_agent = agent_2
        curr_reward = None
        oppo_reward = None
        curr_episode_td_error = agent_1_episode_td_error
        oppo_episode_td_error = agent_2_episode_td_error

        # Turn iteration
        while not done:
            total_steps += 1

            # step
            action = curr_agent.get_action(curr_state)
            next_state, reward, done, info = env.step(action)

            # print step status
            print_step_status(
                curr_agent, curr_state, action, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            # fit reward to agent
            if "1" in curr_agent.name:
                curr_reward = reward
                oppo_reward = -reward
            elif "2" in curr_agent.name:
                curr_reward = -reward
                oppo_reward = reward

            ######################################################################
            if prev_state is not None:
                if isinstance(oppo_agent, (TTTAgentDqn, TTTAgentReinforce, TTTAgentA2C)):
                    oppo_episode_td_error = oppo_agent.learning(
                        prev_state, prev_action, next_state, oppo_reward, done
                    )

            if done:
                if isinstance(curr_agent, (TTTAgentDqn, TTTAgentReinforce, TTTAgentA2C)):
                    curr_episode_td_error = curr_agent.learning(
                        curr_state, action, next_state, curr_reward, done
                    )

                print_game_statistics(
                    info, episode, epsilon, total_steps, game_status, play_type
                )
            #############################################

            prev_action = action
            prev_state = curr_state
            curr_state = next_state

            # change agent_1_2
            if "1" in curr_agent.name:
                curr_agent, oppo_agent = agent_2, agent_1
                agent_1_episode_td_error = curr_episode_td_error
                agent_2_episode_td_error = oppo_episode_td_error
                curr_episode_td_error = agent_2_episode_td_error
                oppo_episode_td_error = agent_1_episode_td_error
            elif "2" in curr_agent.name:
                curr_agent, oppo_agent = agent_1, agent_2
                agent_2_episode_td_error = curr_episode_td_error
                agent_1_episode_td_error = oppo_episode_td_error
                curr_episode_td_error = agent_1_episode_td_error
                oppo_episode_td_error = agent_2_episode_td_error

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)
        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    # Draw performance
    draw_performance(game_status, MAX_EPISODES)

    return agent_1, agent_2


if __name__ == '__main__':
    current = time.strftime('%y-%m-%d/%X', time.localtime(time.time()))
    print(current)
    agent_1, agent_2 = learning_for_agent_1_vs_agent_2()
    print(current)
    print(time.strftime('%y-%m-%d/%X', time.localtime(time.time())))
