import random


class Dummy_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, state):
        available_actions = state.get_available_actions()
        action = random.choice(available_actions)

        return action
