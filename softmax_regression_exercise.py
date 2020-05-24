import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
"""
    生成数据集
    C1 代表的是从高斯分布采样 (X,Y) ~ N(3,6,1,1,0)
    C2 代表的是从高斯分布采样 (X,Y) ~ N(6,3,1,1,0)
    C3 代表的是从高斯分布采样 (X,Y) ~ N(7,7,1,1,0)
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

x_b = np.random.normal(7., 1, dot_num)
y_b = np.random.normal(7., 1, dot_num)
y = np.ones(dot_num)*2
C3 = np.array([x_b, y_b, y]).T

dataset = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(dataset)

#用于可视化数据集分布
plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')
plt.show()

epsilon = 1e-12
class SoftmaxRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.W = nn.Parameter(torch.randn(2,3).uniform_(-0.1,0.1)) #<-填空
        self.b = nn.Parameter(torch.zeros(3)) #<-填空

    def forward(self,inp):
        logit = torch.matmul(inp, self.W) + self.b #shape (N, 3)

        pred = F.softmax(logit, dim = 1)#shape (N, 3)  #<-注意这里已经算了softmax
        #print(pred)
        return pred   


def computer_loss(pred,label):
    one_hot_label = torch.zeros(label.shape[0], 3)
    one_hot_label.scatter_(1, label.unsqueeze(1), 1.0) #onehot label shape (N,3)

    losses = (-1)*(one_hot_label*torch.log(pred))
    loss = torch.mean(losses)
    accuracy = torch.mean(torch.eq(torch.argmax(pred,dim=-1),label).float())
    return loss, accuracy

def train_one_step(model, optimizer, x, y):
    optimizer.zero_grad()
    pred = model(x)
    loss, accuracy = computer_loss(pred, y)
    loss.backward()
    optimizer.step()
    return loss.detach(), accuracy.detach(), model.W.detach(), model.b.detach()
  

if __name__ == '__main__':
    #定义模型与优化器
    model = SoftmaxRegression()
    optimizer = optim.SGD(model.parameters(),lr = 0.01)

    #生成数据集
    x1, x2, y = list(zip(*dataset))
    x = torch.tensor(list(zip(x1, x2)))
    y = torch.tensor(y).long()

    #训练
    for i in range(1000):
        loss, accuracy, W_opt, b_opt = train_one_step(model, optimizer, x, y)
        if i%50 == 49:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')

    #展示结果
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
    plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')
    x = np.arange(0., 10., 0.1)
    y = np.arange(0., 10., 0.1)
    X, Y = np.meshgrid(x, y)
    inp = torch.tensor(list(zip(X.reshape(-1), Y.reshape(-1))))
    Z = model(inp)
    Z = Z.detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z)
    plt.show()