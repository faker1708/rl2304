import mlp_b
import torch

hpl = [4,3,2,1]
hp = dict()
hp['list']=hpl

nn = mlp_b.mlp(hp)
x = torch.normal(0,1,(4,1))

y = nn.test(x)
print(y)