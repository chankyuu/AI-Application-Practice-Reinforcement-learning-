# 선수 에이전트: Dummy 에이전트, 후수 에이전트: RL 에이전트
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tic_tac_toe_343.common.d_utils import model_load
from tic_tac_toe_343.agents.b_human_agent import Human_Agent
from tic_tac_toe_343.common.a_env_tic_tac_toe_343 import TicTacToe343

# from agents.c_dqn_agent import TTTAgentDqn
from tic_tac_toe_343.agents.c_dqn_agent import TTTAgentDqn

# from agents.d_reinforce_agent import TTTAgentReinforce
from tic_tac_toe_343.agents.d_reinforce_agent import TTTAgentReinforce

# from agents.e_a2c_agent import TTTAgentA2C
from tic_tac_toe_343.agents.e_a2c_agent import TTTAgentA2C


def play_with_agent_2(env, agent_2):
    env.print_board_idx()
    state = env.reset()

    agent_1 = Human_Agent(name="AGENT_1", env=env)
    current_agent = agent_1

    print()

    print("[당신(사람) 차례]")
    env.render()

    done = False
    while not done:
        action = current_agent.get_action(state)
        next_state, _, done, info = env.step(action)
        if current_agent == agent_2:
            print("     State:", state)
            # print("   Q-value:", current_agent.get_q_values_for_one_state(state))
            # print("    Policy:", current_agent.get_policy_for_one_state(state))
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "Q-Learning 에이전트 차례" if current_agent == agent_1 \
            else "당신(사람) 차례"
        ))
        env.render()

        if done:
            if info['winner'] == 1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            elif info['winner'] == -1:
                print("Q-Learning 에이전트가 이겼습니다.")
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
    agent_2 = TTTAgentDqn(name="AGENT_2", env=env)
    # agent_1 = TTTAgentReinforce(name="AGENT_2", env=env)
    # agent_1 = TTTAgentA2C(name="AGENT_2", env=env)

    model_file_name = "DQN_BACK_31.0_12010827043.691.pth"
    model_load(agent_2.q, file_name=model_file_name)
    play_with_agent_2(env, agent_2)
