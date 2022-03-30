#!/usr/bin/env python3
import torch
LOAD_MODEL = r'C:\Users\leopc\Desktop\a3c-continuous-action-space-master\a3c-continuous-action-space-master\pong\models\continue_2_467M.pt'
net = torch.load(LOAD_MODEL)
print('Model loaded from:', LOAD_MODEL)