import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
is_cuda = torch.cuda.is_available()

class Policy(nn.Module):
    def __init__(self, num_actions=2):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.affine2 = nn.Linear(200, num_actions) # action 1: static, action 2: move up, action 3: move down
        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []
        rand_var = torch.tensor([0.5,0.5])
        if is_cuda: rand_var = rand_var.cuda()
        self.random = Categorical(rand_var)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda: x = x.cuda()
        probs = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))
        return action


class Policy_cliff(nn.Module):
    def __init__(self, num_actions = 4):
        super(Policy_cliff, self).__init__()
        self.affine1 = nn.Linear(1, 24)
        self.affine2 = nn.Linear(24, 12)
        self.affine3 = nn.Linear(12, num_actions)
        self.num_actions = num_actions

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_probs = F.softmax(self.affine3(x), dim=-1)
        return action_probs

    def select_action(self, state):
        state = Variable(torch.tensor(state).float().unsqueeze(0))
        if is_cuda: state = state.cuda()
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    # def select_action(state):
    #     state = torch.from_numpy(state).float().unsqueeze(0)
    #     probs = policy(state)
    #     m = Categorical(probs)
    #     action = m.sample()
    #     policy.saved_log_probs.append(m.log_prob(action))
    #     return action.item()
