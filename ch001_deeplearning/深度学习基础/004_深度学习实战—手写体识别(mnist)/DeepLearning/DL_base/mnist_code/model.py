#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm_ws 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：Huajie Sun
@Date    ：2023/7/10 下午3:11
@anno    ：This is a file about  
'''
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),  # W=（F-K+2P）/S+1
            torch.nn.ReLU(),  # 针对每个channel进行单独的激活
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        # print("in: ", x.shape)  # torch.Size([32, 1, 28, 28])

        batch_size = x.size(0)

        x = self.conv1(x)  # 卷积+激活+池化
        # print("conv1: ", x.shape)  # torch.Size([32, 10, 12, 12])

        x = self.conv2(x)  # 卷积+激活+池化
        # print("conv2: ", x.shape)  # torch.Size([32, 20, 4, 4])

        x = x.view(batch_size, -1)  # 展平(batch, 20,4,4) ==> (batch, 320)
        # print("view: ", x.shape)  # torch.Size([32, 320])

        x = self.fc(x)  # 全连接输出类别长度
        # print("fc: ", x.shape)  # torch.Size([32, 10])

        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）