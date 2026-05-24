import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL  #

class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if self.kernel_size & 1 == 0:
            front = x[:, 0:1].repeat(1, (self.kernel_size -1 ) // 2)  
            end = x[:, -1:, ].repeat(1, (self.kernel_size) // 2) 
        else:
            front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2) 
            end = x[:, -1:,].repeat(1, (self.kernel_size - 1) // 2) 
        x = torch.cat([front, x, end], dim=1) 
        x = torch.unsqueeze(x, dim=1)
        x = self.avg(x) 
        x = torch.squeeze(x, dim=1)
        return x


class STLDecomp(nn.Module):

    def __init__(self, period, seasonal=7, trend=None, robust=True):  # 添加robust参数
        super(STLDecomp, self).__init__()
        self.period = period  # 季节性周期
        self.seasonal = seasonal  # 季节性平滑窗口（正奇数）
        self.trend = trend  # 趋势平滑窗口（默认自动计算）
        self.robust = robust  # 鲁棒性估计开关

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        seasonal = torch.zeros_like(x)
        trend = torch.zeros_like(x)

        for i in range(batch_size):
            # 提取单条序列（转为numpy）
            series = x[i].cpu().detach().numpy()
            # 初始化STL：将robust参数传入STL构造函数
            stl = STL(
                series,
                period=self.period,
                seasonal=self.seasonal,
                trend=self.trend,
                robust=self.robust  # 正确位置：在STL初始化时传入
            )
            # fit()方法无需传入robust
            result = stl.fit()  # 移除robust参数
            # 提取季节性和趋势成分，转回tensor
            seasonal[i] = torch.tensor(result.seasonal, device=x.device)
            trend[i] = torch.tensor(result.trend, device=x.device)

        return seasonal, trend  # 返回季节性和趋势成分

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


