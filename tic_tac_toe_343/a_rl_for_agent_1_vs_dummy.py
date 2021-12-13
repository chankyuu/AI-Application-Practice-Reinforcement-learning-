# 선수 에이전트: RL 에이전트, 후수 에이전트: Dummy 에이전트
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from agents.b_human_agent import Human_Agent
from tic_tac_toe_343.agents.a_dummy_agent import Dummy_Agent
from tic_tac_toe_343.common.a_env_tic_tac_toe_343 import TicTacToe343
from tic_tac_toe_343.common.c_game_stats import draw_performance, print_game_statistics, \
    epsilon_scheduled, GameStatus
from tic_tac_toe_343.common.d_utils import PLAY_TYPE, EarlyStopModelSaver

# from agents.c_dqn_agent import TTTAgentDqn
from tic_tac_toe_343.agents.c_dqn_agent import TTTAgentDqn

# from agents.d_reinforce_agent import TTTAgentReinforce
from tic_tac_toe_343.agents.d_reinforce_agent import TTTAgentReinforce

# from agents.e_a2c_agent import TTTAgentA2C
from tic_tac_toe_343.agents.e_a2c_agent import TTTAgentA2C

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 100_000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 200_000

STEP_VERBOSE = False
BOARD_RENDER = False


# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Dummy 에이전트
def learning_for_agent_1_vs_dummy():
    # Create environment
    game_status = GameStatus()
    env = TicTacToe343()

    # Create agent
    # agent_1 = TTTAgentDqn(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001,
    #     replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=1_000,
    #     min_buffer_size_for_training=1_000
    # )

    # agent_1 = TTTAgentReinforce(
    #     name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001
    # )

    agent_1 = TTTAgentA2C(
       name="AGENT_1", env=env, gamma=0.99, learning_rate=0.00001, batch_size=32
    )

    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    total_steps = 0

    early_stop_model_saver = EarlyStopModelSaver(target_win_rate=95.0)
    early_stop = False
    win_rate = 0.0
    agent_1_episode_td_error = 0.0

    # Episodes
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False

        agent_1_episode_td_error = 0.0

        # Turns (2 time steps)
        while not done:
            total_steps += 1
            if isinstance(agent_1, TTTAgentDqn): agent_1.time_steps += 1

            # agent_1 스텝 수행
            action = agent_1.get_action(state, epsilon, mode="TRAIN")
            next_state, reward, done, info = env.step(action)

            if done:
                # reward: agent_1이 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error = agent_1.learning(
                    state, action, next_state, reward, done
                )

                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                win_rate = print_game_statistics(
                    info, episode, epsilon, total_steps, game_status, PLAY_TYPE.FIRST
                )
            else:
                # agent_2 스텝 수행
                action_2 = agent_2.get_action(next_state)
                next_state, reward, done, info = env.step(action_2)

                if done:
                    # reward: agent_2가 착수하여 done=True
                    # agent_2가 이기면 -1.0, 비기면 0.0
                    agent_1_episode_td_error = agent_1.learning(
                        state, action, next_state, reward, done
                    )

                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    win_rate = print_game_statistics(
                        info, episode, epsilon, total_steps, game_status, PLAY_TYPE.FIRST
                    )
                else:
                    agent_1_episode_td_error = agent_1.learning(
                        state, action, next_state, reward, done
                    )

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)

        if episode > 5000:
            early_stop = early_stop_model_saver.check(
                agent_1.agent_type, PLAY_TYPE.FIRST, win_rate, agent_1_episode_td_error, agent_1.model
            )
            if early_stop:
                break

    if not early_stop:
        early_stop_model_saver.save_checkpoint(
            agent_1.agent_type, PLAY_TYPE.BACK, win_rate, agent_1_episode_td_error, agent_1.model
        )

    draw_performance(game_status, episode)


if __name__ == '__main__':
    learning_for_agent_1_vs_dummy()
