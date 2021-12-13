# 선수 에이전트: RL 에이전트, 후수 에이전트: RL 에이전트
import os

from tic_tac_toe_343.common.c_game_stats import print_game_statistics, GameStatus
from tic_tac_toe_343.common.a_env_tic_tac_toe_343 import TicTacToe343
from tic_tac_toe_343.common.d_utils import MODEL_DIR, PLAY_TYPE, model_load

# from agents.c_dqn_agent import TTTAgentDqn
from tic_tac_toe_343.agents.c_dqn_agent import TTTAgentDqn

# from agents.d_reinforce_agent import TTTAgentReinforce
from tic_tac_toe_343.agents.d_reinforce_agent import TTTAgentReinforce

# from agents.e_a2c_agent import TTTAgentA2C
from tic_tac_toe_343.agents.e_a2c_agent import TTTAgentA2C


def self_play(env, self_agent):
    MAX_EPISODES = 10000
    VERBOSE = False

    agent_1 = self_agent
    agent_2 = self_agent

    current_agent = agent_1

    game_status = GameStatus()
    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        if VERBOSE:
            print("[시작 상태]")
            env.render()

        done = False
        while not done:
            total_steps += 1
            action = current_agent.get_action(state)
            next_state, _, done, info = env.step(action)

            if VERBOSE:
                print("[{0}]".format("Q-Learning 에이전트 1" if current_agent == agent_1 else "Q-Learning 에이전트 2"))
                env.render()

            if done:
                if VERBOSE:
                    if info['winner'] == 1:
                        print("Q-Learning 에이전트 1이 이겼습니다.")
                    elif info['winner'] == -1:
                        print("Q-Learning 에이전트 2가 이겼습니다!")
                    else:
                        print("비겼습니다!")

                done = done
                print_game_statistics(info, episode, 0.0, total_steps, game_status)
            else:
                state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1


if __name__ == '__main__':
    env = TicTacToe343()
    agent = TTTAgentDqn(name="SELF_AGENT", env=env)
    # agent = TTTAgentReinforce(name="SELF_AGENT", env=env)
    # agent = TTTAgentA2C(name="SELF_AGENT", env=env)

    model_file_name = os.path.join(MODEL_DIR, "{0}_{1}.pth".format(
        agent.agent_type, PLAY_TYPE.BACK.value
    ))
    model_load(agent.model, file_name=model_file_name)
    self_play(env, agent)
