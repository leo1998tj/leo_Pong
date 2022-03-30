from Utils.Fi_score import calculate_FI
from Utils.Image_pro import get_prepro_state
from Models.A3c_model_for_test import ModelA3C

import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import gym


model_path = '..\Weights\pg_params.pkl'
is_train = False
states_data_path = r'E:\code\states'
test_states_data_path = r'E:\code\states\test'
#最终储存结果
res_path = r'C:\Users\leopc\Desktop\leo_Pong\Datas\PG'

if __name__ =='__main__':
    env = gym.make('Pong-v0')
    obs_shape = env.observation_space.shape
    print('Observation shape:', obs_shape)
    act_space = env.action_space.n
    print('Action space:', act_space)
    # built policy network
    policy = ModelA3C(obs_shape, act_space)
    policy.load_state_dict(torch.load(model_path))
    model = policy.cuda()

    # get Data
    if is_train:
        is_train = "train"
    else:
        is_train = "test"
    states_data_path = r"{}\{}".format(states_data_path,is_train)
    files = os.listdir(states_data_path)
    for file in files:
        print("begin: {}".format(file))
        path = "{}\{}".format(states_data_path, file)
        final_path = ""
        final_path = "{}\{}\{}.npy".format(res_path, is_train, file.split(".")[0])
        datas = np.load(path)
        datas = get_prepro_state(datas)
        res = []
        for i in range(len(datas) - 1):
            data = datas[i + 1] - datas[i]
            FI = calculate_FI(data, model)
            res.append(FI)
        np.save(final_path, res)
        print("{} is success".format(final_path))