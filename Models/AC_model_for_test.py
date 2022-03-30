import argparse
import os
from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
is_cuda = torch.cuda.is_available()
class AC(nn.Module):
    def __init__(self, num_actions=2):
        super(AC, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.action_head = nn.Linear(200, num_actions)  # action 1: static, action 2: move up, action 3: move down
        self.value_head = nn.Linear(200, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda: x = x.cuda()
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs[-1].append((m.log_prob(action), state_value))
        return action