import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
    生成数据集
    C1 代表从高斯分布采样 （X，Y）~ N（3,6,1,1,0）
    C2 代表从高斯分布采样 （X，Y）~ N（6,3,1,1,0）
"""

dot_num = 100
x_p = np.random.normal(3., 1, dot_num)
y_p = np.random.normal(6., 1, dot_num)
y = np.ones(dot_num)
C1 = np.array([x_p, y_p, y]).T

x_n = np.random.normal(6., 1, dot_num)
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)
C2 = np.array([x_n, y_n, y]).T

#将C1 C2连接成一个新的数组
dataset = np.concatenate((C1, C2), axis=0)
#以dataset 生成随机数组
np.random.shuffle(dataset)

"""
#可视化数据集分布
plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
plt.show()
"""

"""
    建立模型
    实现建立逻辑回归模型，定义loss函数以及单步梯度下降过程函数
"""
epsilon = 1e-12
class LogisticRegression(nn.Module):
    def __init__(self):
        #调用父类 nn.Module.__init__()函数
        super(LogisticRegression,self).__init__()
        #将固定的不可训练的tensor 转化为一个可以训练的类型的parameter，且将在这个parameter绑定到这个module里
        #在参数优化时可以进行优化,初始化参数W及b
        self.W = nn.Parameter(torch.randn(2,1).uniform_(-0.1,0.1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, inp):
        #线性的判别函数 f(x;w) = wx + b
        logit = torch.matmul(inp,self.W) + self.b #shape (N,1)
        pred = torch.sigmoid(logit)
        #print("hhh,",pred.shape())
        return pred


def computer_loss(pred, label):
    #去掉pred中dim = 1 的 维数为1  
    pred = torch.squeeze(pred, dim = 1)

    #losses = nn.CrossEntropyLoss(pred, label)
    losses = (label*torch.log(pred) + (1-label)*torch.log(1-pred))*(-1)
    loss = torch.mean(losses)

    #大于0.5时 pred =1 , <0.5时：pred = 0， 确定其标签
    pred = torch.where( pred > 0.5, torch.ones_like(pred),torch.zeros_like(pred))
    accuracy = torch.mean( torch.eq(pred,label).float() )
    return loss, accuracy

def train_one_step(model, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss, accuracy = computer_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach(), accuracy.detach(), model.W.detach(), model.b.detach()

if __name__ == '__main__':
    #定义模型和优化器
    model = LogisticRegression()
    #创建一个学习率为0.01的优化器    
    optimizer = optim.SGD(model.parameters(),lr = 0.01)

    #生成数据集
    x1, x2, y = list(zip(*dataset))  #解压 返回三维矩阵形式
    x = torch.tensor(list(zip(x1, x2)))  #压缩
    y = torch.tensor(y)

    #训练
    animation_fram = []
    for i in range(400):
        loss, accuracy, W_opt, b_opt = train_one_step(model, optimizer, x, y)
        animation_fram.append((W_opt.numpy()[0,0], W_opt.numpy()[1,0],b_opt.numpy(),loss.numpy()))
        if i%20 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')

    #展示结果 version1
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    W_1 = model.W[0].item()
    W_2 = model.W[1].item()
    b = model.b[0].item()
    xx = np.arange(10, step=0.1)
    yy =  - W_1 / W_2 * xx - b / W_2
    plt.plot(xx,yy)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()        