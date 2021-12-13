import enum
import numpy as np
import torch
import os
from glob import glob

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class AGENT_TYPE(enum.Enum):
    DQN = "DQN"
    REINFORCE = "REINFORCE"
    A2C = "A2C"


class PLAY_TYPE(enum.Enum):
    FIRST = "FIRST"
    BACK = "BACK"
    SELF = "SELF"


def model_load(model, file_name):
    model_params = torch.load(os.path.join(MODEL_DIR, file_name))
    model.load_state_dict(model_params)


class EarlyStopModelSaver:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=30, target_win_rate=95.0):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간 (Default: 7)
        """
        self.patience = patience
        self.counter = 0
        self.min_loss = np.inf
        self.target_win_rate = target_win_rate

    def check(self, agent_type, play_type, win_rate, loss, model):
        early_stop = False

        if win_rate >= self.target_win_rate:
            if self.min_loss == np.inf:
                '''최초 모델을 저장한다.'''
                self.save_checkpoint(agent_type, play_type, win_rate, loss, model)
                self.min_loss = loss
                self.target_win_rate = win_rate
            elif loss >= self.min_loss:
                self.counter += 1
                #print(f'counter: {self.counter} out of {self.patience} - {loss}')
                if self.counter >= self.patience:
                    early_stop = True
                    print("EARLY STOP!")
            else:
                '''loss가 감소하면 모델을 저장한다.'''
                self.save_checkpoint(agent_type, play_type, win_rate, loss, model)
                self.min_loss = loss
                self.counter = 0

        return early_stop

    def save_checkpoint(self, agent_type, play_type, win_rate, loss, model):
        target_remove_files = glob(os.path.join(MODEL_DIR, "{0}_{1}_*.pth".format(
            agent_type, play_type.value
        )))
        for file_name in target_remove_files:
            os.remove(file_name)

        model_file_name = os.path.join(MODEL_DIR, "{0}_{1}_{2:.1f}.pth".format(
            agent_type, play_type.value, win_rate
        ))

        print(
            f'Win Rate is {win_rate:.2f} and Loss decreased ({self.min_loss:.6f} --> {loss:.6f}).  '
            f'Saving model tp {model_file_name}'
        )

        torch.save(model.state_dict(), model_file_name)
        self.min_loss = loss
