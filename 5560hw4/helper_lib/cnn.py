# helper_lib/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的 CNN 定义复制过来