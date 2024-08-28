#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm_ws 
@File    ：infer.py
@IDE     ：PyCharm 
@Author  ：Huajie Sun
@Date    ：2023/7/19 上午11:28
@anno    ：This is a file about 推理
'''
import cv2
import torch
from PIL import Image
from torchvision.transforms import transforms

# from model import Net

# device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

def get_data(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # 将数据转为pytorch数据类类型（tensor）
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)

    # 转换为灰度图（因为mnist数据集数据都是灰度图）
    gray_image = image.convert('L')

    gray_image = transform(gray_image)
    gray_image = gray_image.unsqueeze(0)

    return gray_image
    
if __name__ == '__main__':
    # 1，model define
    model = Net().to(device)
    checkpoints = "checkpoint_mnist.pth"
    model.load_state_dict(torch.load(checkpoints))
    model.eval()

    # data
    image_path = "./test.png"
    image = get_data(image_path)

    # infer
    outputs = model(image.to(device))
    _, predicted = torch.max(outputs.data, dim=1)
    print(predicted.data)