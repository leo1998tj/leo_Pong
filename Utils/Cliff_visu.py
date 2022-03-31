from pandas import DataFrame
from scipy.stats import uniform
from scipy.stats import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Models.Policy_model_for_test import Policy_cliff
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
from gym.envs.toy_text import CliffWalkingEnv
from torch import optim
from collections import deque

# path = "..//Envs//poltting_data.npy"
# data = np.load(path)
# plt.plot(data)
# plt.show()

env = CliffWalkingEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Policy_cliff(num_actions=env.action_space.n).to(device)

model.load_state_dict(torch.load(r"D:\Rl\leo_pong\Envs\daliy.pt"))
model = model.to(device)

for i in range(48):
    print(i, model.get_pro(i))
