#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pycharm_ws 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：Huajie Sun
@Date    ：2023/7/10 下午3:13
@anno    ：This is a file about  
'''
import torch
import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from model import Net

# device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

def get_data(root, batch_size, show=False):

    transform = transforms.Compose([
        transforms.ToTensor(),  # 将数据转为pytorch数据类类型（tensor）
        transforms.Normalize((0.1307,), (0.3081,))
        # 图像归一化：对图像像素值进行预处理，目的是将图像数据缩放到合适的范围或分布。
        #   这样在预测的时候也会把输入的图像按照这个参数进行归一化，防止一些奇怪的数据影响预测准确率。
        #   提高模型的泛化能力。
        # 归一化的方法：
        #   1，最大最小值归一化：将像素值线性缩放到指定的范围，如[0, 1]或[-1, 1]。
        #   2，均值方差归一化：将像素值减去均值，并除以标准差，使得数据分布具有零均值和单位方差。
    ])

    # 一般这一步是需要自己根据实际数据集定义
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform,
                                   download=True)  # 本地没有就加上download=True
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform,
                                  download=True)  # train=True训练集，=False测试集

    # 固定
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if show:
        fig = plt.figure()
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
            plt.title("Labels: {}".format(train_dataset.train_labels[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0  # 处理了多少样本
    running_correct = 0  # 正确预测的样本

    train_loader = tqdm.tqdm(train_loader)
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs.to(device))  # 推理

        # 损失计算
        # print(outputs.shape, target.shape)  # torch.Size([32, 10]) torch.Size([32])
        loss = criterion(outputs, target.to(device))

        loss.backward()  # 损失后向传播（当前损失对所有节点求导）
        optimizer.step()  # 梯度更新（使用loss对每个节点计算的梯度进行每个结点的参数更新）

        running_loss += loss.item()  # 累加当前epoch的loss

        _, predicted = torch.max(outputs.data, dim=1)  # 预测最大概率

        # 统计
        running_total += inputs.shape[0]  # 总计数量
        running_correct += (predicted == target.to(device)).sum().item()  # 正确预测数量

    acc = running_correct / running_total
    print("train acc: ", acc)
    
    torch.save(model.state_dict(), "checkpoint_mnist.pth")

    return acc

def test_one_epoch(model, test_loader, epoch):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    acc = correct / total
    print("test acc: ", acc)

    return acc

def show_acc_fun(train_acc_list, test_acc_list, epochs):
    # 创建数据
    epoch_list = [e for e in range(epochs)]

    # 画图
    plt.plot(epoch_list, train_acc_list, label="train acc")
    plt.plot(epoch_list, test_acc_list, label="test acc")

    # 添加标题和标签
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('acc')

    # 显示图像
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # 1，model define
    model = Net().to(device)

    # 2，data
    root = "./data/mnist"
    batch_size = 32
    train_loader, test_loader = get_data(root, batch_size, show=False)

    # loss
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    """  适用于分类问题：衡量了预测概率分布与真实概率分布之间的差异
        1，预测与label数据维度（都是概率）： out: torch.Size([B, 10]) label: torch.Size([32])
        2，CEL使用对数损失进行计算，假设有一个包含C(10)个类别的分类问题，并且对于某个样本，真实类别标签概率用yi表示
          （其中i从1到C变化）。每个类别的预测概率由pi表示（同样i从1到C变化），公式如下：
          L = -∑(yi * log(pi))
          求和是针对所有类别进行的，损失值L对错误的预测给予惩罚，当对于真实类别的预测概率较低时，损失值较高。
    """

    # optimizer
    learning_rate = 0.01
    momentum = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum)  # lr学习率，momentum冲量

    # train
    show_acc = True
    epochs = 5
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epochs):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        train_acc_list.append(train_acc)

        test_acc = test_one_epoch(model, test_loader, epoch)
        test_acc_list.append(test_acc)

    if show_acc:
        show_acc_fun(train_acc_list, test_acc_list, epochs)