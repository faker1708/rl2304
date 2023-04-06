
import torch
# cat 是合并，维度不变
# stack 是升维

# 一般用cat


# a = torch.tensor([[1],[2]])
# b = torch.tensor([[3],[4]])
# print(a)

# # c = torch.stack((a))
# # print(c)
# c = torch.stack((a,b),0)
# print(c)


# c = torch.stack((a,b),1)
# print(c)


# c = torch.stack((a,b),2)
# print(c)


# c = torch.cat((a,b),1)
# print(c)


# a = torch.tensor([[1],[2]])
# c = torch.tensor([1])
# b = a.gather(1,c)
# print(b)
# import torch

tensor_0 = torch.arange(3, 12).view(3, 3)
print(tensor_0)

# index = torch.tensor([[2, 1, 0]])
index = torch.tensor([[1, 0, 0]])
tensor_1 = tensor_0.gather(0, index)
print(tensor_1)


xxx = torch.tensor([[1.7837, 3.6783, 6.4240, 9.0323],
                    [7.2929, 7.7416, 8.7126, 9.5165]], device='cuda:0')
ac = torch.tensor([[1., 1., 1., 1.]], device='cuda:0')
ac = ac.long()
# ac = torch.Tensor(ac).cuda()
# ac = ac.item()
print(xxx,ac)
print(ac.dtype)
# ac = torch.tensor([[1,0,0,1]]).cuda()

b= xxx.gather(0,ac)

print(b)