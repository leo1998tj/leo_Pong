import pandas as pd
import numpy as np
import os

data_path = r"C:\Users\leopc\Desktop\leo_Pong\Datas"

# 获取打分好的数组
# get_FI_score("PG")
def get_FI_score(model_name, is_Train = True):
    sub_file = "train"
    if not is_Train: sub_file = "test"
    path = "{}/{}/{}".format(data_path, model_name,sub_file)
    files = os.listdir(path)
    res =np.array([])
    for file in files:
        single_path = "{}\{}".format(path, file)
        datas = np.load(single_path)
        res = np.append(res, datas)
    return res
    print(res.shape)
