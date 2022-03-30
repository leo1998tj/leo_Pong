import gym

env=gym.make("CliffWalking-v0") #创建对应的游戏环境
env.seed(1) #可选，设置随机数，以便让过程重现
env=env.unwrapped #可选，为环境增加限制，对训练有利

#----------------------动作空间和状态空间---------------------#
print(env.action_space) #动作空间，输出的内容看不懂              Discrete(4)
print(env.action_space.n) #当动作是离散的时，用该方法获取有多少个动作  4
# env.observation_space.shape[0] #当动作是连续的时，用该方法获取动作由几个数来表示
print(env.action_space.sample()) #从动作空间中随机选取一个动作   1
# 同理，还有 env.observation_space，也具有同样的属性和方法（.low和.high方法除外）
#-------------------------------------------------------------#


# 0 是向上， 1是向右， 2是向下， 3是向左
flag = False
actions = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2] # 最好成绩应该是-13
for i in actions:
    if flag:
        break
    else:
        env.render()
        a = i
        s_, r, done, info = env.step(a)
        print(s_)
        print(r)
        print(done)
        print(info)
        env.render()
# for episode in range(100): #每个回合
#     s=env.reset() #重新设置环境，并得到初始状态
#     while True: #每个步骤
#         env.render() #展示环境
#         a=env.action_space.sample() # 智能体随机选择一个动作
#         s_,r,done,info=env.step(a) #环境返回执行动作a后的下一个状态、奖励值、是否终止以及其他信息
#         if done:
#             break
