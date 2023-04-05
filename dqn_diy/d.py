

import mlp

import torch


sp = [4,3,2,1]
lr= 0.03
batch_size = 7

aa = mlp.mlp(sp,lr,batch_size)

x = torch.normal(0,1,(4,batch_size)).cuda()
y = aa.test(x)
print(x,y)


