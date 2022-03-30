import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import scipy.linalg as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
is_cuda = torch.cuda.is_available()

# 计算Fi分数的函数
def calculate_FI(model_input, model, epsilon = 1e-8):
    states = Variable(model_input, requires_grad = True)
    prob = model(states)
    if len(prob) > 1:
        prob = prob[0]
    a = torch.argmax(prob, axis=1)
    dist = Categorical(prob)
    log_prob = dist.log_prob(a)
    log_prob2 = torch.log(prob[:,0])
    log_prob3 = torch.log(prob[:,1])

    loss = torch.sum(log_prob)
    loss2 = torch.sum(log_prob2)
    loss3 = torch.sum(log_prob3)

    loss.backward(retain_graph = True)
    grad = states.grad.cpu().numpy().copy()
    #梯度清零
    states.grad.zero_()


    loss2.backward(retain_graph = True)
    grad2 = states.grad.cpu().numpy().copy()
    states.grad.zero_()

    loss3.backward()
    grad3 = states.grad.cpu().numpy().copy()
    states.grad.zero_()

    grad_mat = np.array([grad2, grad3])

    logits = prob.detach().cpu().numpy()
    grad_mat = np.array([grad2, grad3])
    res = []
    for i in range(len(grad)):
        # 对应的概率
        L0 = grad_mat[:, i, :].T * (logits[i] ** 0.5)
        B0, D_L0, A0 = sp.svd(L0, full_matrices=False)
        rank_L0 = sum(D_L0 > epsilon)
        if rank_L0 > 0:
            B0 = B0[:, :rank_L0]
            A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0, :]

            U_A, D_0, _ = sp.svd(A0 @ A0.T, full_matrices=True)
            D_0_inv = np.diag(D_0 ** -1)
            D_0_inv_sqrt = np.diag(D_0 ** -0.5)
            U_0 = B0 @ U_A

            nabla_f = grad[i] @ U_0 @ D_0_inv_sqrt
            FI = nabla_f @ nabla_f.T
            res.append(FI)
        res = np.array(res)

    return res

