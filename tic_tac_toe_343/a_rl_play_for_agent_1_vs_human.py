# 선수 에이전트: RL 에이전트, 후수 에이전트: Dummy 에이전트
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tic_tac_toe_343.agents.b_human_agent import Human_Agent
from tic_tac_toe_343.common.a_env_tic_tac_toe_343 import TicTacToe343
from tic_tac_toe_343.common.d_utils import model_load

# from agents.c_dqn_agent import TTTAgentDqn
from tic_tac_toe_343.agents.c_dqn_agent import TTTAgentDqn

# from agents.d_reinforce_agent import TTTAgentReinforce
from tic_tac_toe_343.agents.d_reinforce_agent import TTTAgentReinforce

# from agents.e_a2c_agent import TTTAgentA2C
from tic_tac_toe_343.agents.e_a2c_agent import TTTAgentA2C


def play_with_agent_1(env, agent_1):
    env.print_board_idx()
    state = env.reset()

    agent_2 = Human_Agent(name="AGENT_2", env=env)

    current_agent = agent_1

    print()

    print("[RL 에이전트 차례]")
    env.render()

    done = False
    while not done:
        if isinstance(current_agent, (TTTAgentDqn, TTTAgentReinforce, TTTAgentA2C)):
            action = current_agent.get_action(state, mode="PLAY")
        else:
            action = current_agent.get_action(state)

        next_state, _, done, info = env.step(action)
        if current_agent == agent_1:
            print("     State:", state)
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "당신(사람) 차례" if current_agent == agent_1 \
            else "RL 에이전트 차례"
        ))
        env.render()

        if done:
            if info['winner'] == 1:
                print("RL 에이전트가 이겼습니다.")
            elif info['winner'] == -1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent = agent_1


if __name__ == '__main__':
    env = TicTacToe343()

    #agent_1 = TTTAgentDqn(name="AGENT_1", env=env)
    #agent_1 = TTTAgentReinforce(name="AGENT_1", env=env)
    agent_1 = TTTAgentA2C(name="AGENT_1", env=env)

    model_file_name = "A2C_FIRST_97.0.pth"
    model_load(agent_1.model, file_name=model_file_name)

    play_with_agent_1(env, agent_1)
