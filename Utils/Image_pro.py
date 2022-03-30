import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
is_cuda = torch.cuda.is_available()
import numpy as np
import heapq
from PIL import Image

# datas = np.load('3800_state_fi.npy', allow_pickle=True)
# save_path = r'C:\Users\leopc\Desktop\pong-master\3800_state.npy'
# image_datas = np.load(save_path)

# 计算分位数
def cal_per(data, k, m):
    res = []
    for i in range(k):
        res.append(np.around(np.percentile(data, i/k), m))
    res.append(np.around(max(data), m))
    return res

# 求出最高的k个数值
def get_top_K(data, k):
    return heapq.nlargest(k, data)

# 求出最高的k个数值并作图
def get_top_K_index(data, k):
    return heapq.nlargest(k, range(len(data)), data.__getitem__)

# 绘制图像进行对比
x = 210
y = 160
def a2i(indata):
    mg = Image.new('L',indata.transpose().shape)
    mn = indata.min()
    a = indata-mn
    mx = a.max()
    a = a*256./mx
    mg.putdata(a.ravel())
    return mg

def Cube2Image(data):
    ri = a2i(data[0])
    gi = a2i(data[1])
    bi = a2i(data[2])
    mg = Image.merge('RGB',(ri,gi,bi))
    return mg

#
# res = get_top_K_index(datas, 5)
# # res = cal_per(datas, 20)
# # image_datas_list = []
# # for index in res:
# #     image_datas_list.append([image_datas[index], image_datas[index+1]])
# #     pic = Image.fromarray(image_datas[index])
# #     pic.save("picture_0321//{}_pic.png".format(index))
# #     pic = Image.fromarray(image_datas[index+1])
# #     pic.save("picture_0321//{}_pic_next.png".format(index))
#
# # [4.92e-06, 1.487e-05, 2.096e-05, 3.107e-05, 4.026e-05, 4.894e-05, 5.562e-05,
# # 6.117e-05, 7.088e-05, 7.776e-05, 8.078e-05, 9.081e-05, 9.236e-05, 9.662e-05,
# # 0.00010179, 0.00011195, 0.00012032, 0.00013128, 0.0001364, 0.00015698, array([16.662512], dtype=float32)]
# print(cal_per(datas, 20, 8))
# # [array([16.662512], dtype=float32), array([12.583109], dtype=float32), array([12.428136], dtype=float32),
# # array([12.028114], dtype=float32), array([11.991705], dtype=float32), array([11.842478], dtype=float32),
# # array([11.585517], dtype=float32), array([11.39511], dtype=float32), array([10.42517], dtype=float32), array([9.733502], dtype=float32)]
# print(get_top_K(datas, 10))
# # [269, 3497, 1425, 1598, 4412, 996, 101, 6909, 2789, 2522]
# print(get_top_K_index(datas, 10))

# 压缩pong图像为80*80
def get_prepro_state(datas):
    res = []
    for s in datas:
        s = 0.2126 * s[:, :, 0] + 0.7152 * s[:, :, 1] + 0.0722 * s[:, :, 2]
        s = s.astype(np.uint8)
        # s = imresize(s, (80, 80)).ravel()
        s = np.array(Image.fromarray(s).resize((80, 80))).ravel()
        res.append(torch.tensor(s, dtype=torch.float).unsqueeze(0).to('cuda'))
    return res# Return a tensor