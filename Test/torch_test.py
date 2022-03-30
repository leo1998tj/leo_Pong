import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque
from gym.envs.toy_text import CliffWalkingEnv
from Models.Policy_model_for_test import Policy_cliff

# A = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(A)
# print(target)
#
# neg_log_prob = F.cross_entropy(input=A, target=target,
#                                reduction='none')
# print(neg_log_prob)

# x1
# x1 = torch.tensor([[11,21,31],[21,31,41]],dtype=torch.int)
# x1.shape # torch.Size([2, 3])
# # x2
# x2 = torch.tensor([[12,22,32],[22,32,42]],dtype=torch.int)
# x2.shape  # torch.Size([2, 3])
#
# 'inputs为２个形状为[2 , 3]的矩阵 '
# inputs = [x1, x2]
# print(inputs)
# '打印查看'
#
# print(torch.cat(inputs, dim=0))
# print(torch.cat(inputs, dim=1))
# print(torch.cat(inputs, dim=-1))
# print(torch.cat(inputs, dim=-2))

objv_all = torch.tensor(0.1)
objv_all2 = torch.tensor(0.2)
sum = objv_all+objv_all2
print(sum)
