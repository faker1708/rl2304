# 封装一下mlp类，它的nn不好用。麻烦死了。


import torch
import torch.nn as nn
import torch.nn.functional as F
class mlp(nn.Module):
    def __init__(self, hyper_parameter):
        self.hyper_parameter = hyper_parameter
        lr= 0.03

        super(mlp, self).__init__()



        # 根据给定的超参列表来构建网络结构。
        hpl = hyper_parameter['list']

        len_hpl = len(hpl)
        matrix_quantity = len_hpl-1

        

        # 64 32
        self.net = list()

        for i in range(matrix_quantity):
            now = hpl[i]
            next = hpl[i+1]
            fca = nn.Linear(now, next)
            fca.weight.data.normal_(0, 0.1)
            self.net.append(fca)
        


        # self.fc1 = nn.Linear(N_STATES, 50)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        # self.out = nn.Linear(50, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)   # initialization

        # self.learn_rate = lr
        # self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.loss_func = nn.MSELoss()
    
    def forward(self, x):

        hpl = self.hyper_parameter['list']

        len_hpl = len(hpl)
        matrix_quantity = len_hpl-1


        for i in range(matrix_quantity):
            fca = self.net[i]
            x = fca(x)
            x = F.relu(x)
        return x
    
    def update(self,q_eval,q_target):
        # lr = self.learn_rate
        
        loss = self.loss_func(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self,x):
        # 关闭梯度更新


        hpl = self.hyper_parameter['list']

        len_hpl = len(hpl)
        matrix_quantity = len_hpl-1

        with torch.no_grad():
            for i in range(matrix_quantity):
                fca = self.net[i]
                x = fca(x)
                x = F.relu(x)
            return x