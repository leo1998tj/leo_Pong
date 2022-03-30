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

from Models.Policy_model_for_test import Policy_cliff

# Hyper Parameters for PG Network
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled = False  # 非确定性算法


class PG(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space
        self.action_dim = env.action_space.n

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = Policy_cliff(num_actions=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            prob_weights = network_output.cuda().data.cpu().numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()

        # print(prob_weights)
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    # 将状态，动作，奖励这一个transition保存到三个列表中
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        self.time_step += 1

        # Step 1: 计算每一步的状态价值
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs = discounted_ep_rs.astype(np.float64)
        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # Step 2: 前向传播
        # print(len(self.ep_obs))
        # print(self.ep_obs)
        softmax_inputs = []
        for sing_ob in self.ep_obs:
            softmax_input = self.network.forward(torch.FloatTensor(sing_ob).to(device))
            softmax_inputs.append(softmax_input.detach().cpu().numpy())
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        softmax_inputs = torch.tensor(softmax_inputs)
        neg_log_prob = F.cross_entropy(input=softmax_inputs.to(device), target=torch.LongTensor(self.ep_as).to(device),
                                       reduction='none')

        # Step 3: 反向传播

        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 3000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode
gamma = 0.99


# initialize OpenAI Gym env and dqn agent
env = CliffWalkingEnv()
agent = PG(env)
optimizer = optim.Adam(agent.network.parameters(), lr=1e-2)
eps = np.finfo(np.float64).eps.item()


def main():
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        for step in range(STEP):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)  # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                break
        agent.learn()  # 更新策略网络
        # Test every 100 episodes
        if episode % 10 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in agent.network.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    # print(returns.shape)
    # print(len(agent.network.saved_log_probs))
    # print(agent.network.saved_log_probs[0].item())
    for log_prob, R in zip(agent.network.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()


    # policy_loss = policy_loss.cpu()

    # print(policy_loss)
    # policy_loss = torch.cat(policy_loss).sum()
    final_loss = policy_loss[0]
    for loss in policy_loss:
        # print(final_loss)
        # print(loss)
        final_loss += loss
    final_loss = final_loss - policy_loss[0]
    final_loss.backward()
    optimizer.step()
    del agent.network.rewards[:]
    del agent.network.saved_log_probs[:]

def main_new():
    # 初始reward
    running_reward = deque()
    for episode in range(EPISODE):
        # initialize task
        state, episode_reward = env.reset(), 0  # episode_reward 用于判断何时停止训练
        for step in range(STEP):
            action = agent.network.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)  # 新函数 存取这个transition
            agent.network.rewards.append(reward)
            episode_reward += reward
            # state = next_state
            if done:
                # print("stick for ",step, " steps")
                break
        if len(running_reward) >= 10:
            running_reward.popleft()
        running_reward.append(episode_reward)
        finish_episode() # 使用新的方法更新参数
        # agent.learn()  # 更新策略网络

        if episode % 10 == 0:
            print('episode: ', episode, 'now reward:', episode_reward, 'Evaluation Average Reward:', np.mean(running_reward))


        if np.mean(running_reward) > -15:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, episode))
            torch.save(agent.network.state_dict(), 'hello.pt')
            break

if __name__ == '__main__':
    time_start = time.time()
    # main()
    main_new()
    time_end = time.time()
    print('The total time is ', time_end - time_start)