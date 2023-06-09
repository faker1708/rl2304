# 使用了神经网络的q_lean算法

import mlp
import random
import copy
import torch


class dqn(object):
    def __init__(self,sp,lr,batch_size):
        self.learn_rate = lr
        self.batch_size = batch_size

        self.capacity = 2**10   # 内存容量
        
        self.memory_pointer = 0 # 内存指针
        self.memory=list()
        for i in range(self.capacity):
            self.memory.append(list())

        self.protagonist_net  = mlp.mlp(sp,lr)
        self.assistant_net = copy.deepcopy(self.protagonist_net)

        self.sync_step = 0
        self.epsilon = 0.9
        # self.epsilon = 1

    def action_f(self,state):
        ep = self.epsilon

        fc = 0  # 服从神经网络的决策
        if(ep==1):
            fc = 1
        else:
            xxc = 2**12
            xxd = int(ep*xxc)
            xxe = random.randint(0,xxc)
            if(xxe<xxd):
                fc = 1
            else:
                fc = 0


        if(fc==1):
            x = torch.from_numpy(state).cuda()
            x = torch.unsqueeze(x, dim=1)
            y = self.protagonist_net.test(x)

            action = int(y.argmax().cpu())
            # print(y.argmax())
            # print(action)


            # print('y',y)
            # print('x',x)
            # action = int(y.cpu())
        else:
            action = self.random_action()

        return action
    def random_action(self):
        # 用来探索
        ac = random.randint(0,1)
        return ac

    def store(self,packet):
    # def store(self,state, action, next_state, reward):
        # 存储一些经验
        # self.memory.append(packet)
        
        #　 指针循环存储数据
        
        smp = self.memory_pointer

        # print(smp,self.capacity)

        if(smp>=self.capacity):smp = 0
        self.memory[smp]=packet

        
        # if(smp>=self.capacity):
        #     smp = 0
        # else:
        #     smp+=1
        smp+=1

        
        self.memory_pointer = smp


        return 0
    
    # def 

    def get_batch(self,batch):
        # 生成四个tensor
        # 每个的列数都是 batch_size
        sl = list()
        # batch_state = torch.normal(0,1,(4,1))
        # batch_state = torch.ones((4,1))
        batch_state = torch.tensor([]).cuda()   # 这个好啊。
        batch_action = torch.tensor([]).cuda()
        batch_next_state = torch.tensor([]).cuda()
        batch_reward = torch.tensor([]).cuda()
        # print(batch)
        # print(len(batch))
        for i,ele in enumerate(batch):
            if(ele):
                state = ele[0]
                tens = torch.from_numpy(state).cuda()
                tens = torch.unsqueeze(tens, dim=1)
                batch_state = torch.cat((batch_state,tens),1)
                
                action = ele[1]
                tt = torch.tensor(action).reshape([1]).cuda()
                # print(tt)
                tt = torch.unsqueeze(tt, dim=1)
                batch_action = torch.cat((batch_action,tt),1)

                # next_state = ele[2]
                # tt = torch.from_numpy(next_state)
                # print(tt)
                # tt = torch.unsqueeze(tt, dim=1)
                # batch_next_state = torch.cat((batch_action,tt),1)

                
                next_state = ele[2]
                # print(next_state)
                next_state = torch.from_numpy(next_state).cuda()
                # print('next_state',next_state)
                next_state = torch.unsqueeze(next_state, dim=1)
                batch_next_state = torch.cat((batch_next_state,next_state),1)



                reward = ele[3]
                tt = torch.tensor(reward).reshape([1]).cuda()
                tt = torch.unsqueeze(tt, dim=1)
                batch_reward = torch.cat((batch_reward,tt),1)


        # print(batch_state)
        # print(batch_action)
        # print(batch_next_state)
        # batch_state = torch.tensor(sl)
        return batch_state,batch_action,batch_next_state,batch_reward

    def learn(self):
        
        # 随机取出一个批量
        ri = random.randint(0,self.capacity-self.batch_size)
        # ri = 0
        batch = self.memory[ri:ri+self.batch_size]

        for i,ele in enumerate(batch):
            if(ele):
                pass
            else:
                # pass
                return
        
        # print('可以学习了')
        # exit()

        # print(batch)

        beta = 0.9  # 即延系数
        alpha = 0.9 # 先后系数
        alpha = 1

        # 现在要把列表转成tensor 来批量运算。
        

        batch_state,batch_action,batch_next_state,batch_reward  = self.get_batch(batch)

        # print(batch_next_state)

        instant_gratification = batch_reward
        deferred_gratification = self.assistant_net.test(batch_next_state) # 只计算，不反向传播

        # print('deferred_gratification',deferred_gratification)

        total_reward = (1-beta) * instant_gratification + beta* deferred_gratification
        
        
        posterior_reward = total_reward
        priori_reward = self.protagonist_net.forward(batch_state) #需要反向传播
        ideal = (1-alpha)*priori_reward + alpha* posterior_reward



        loss = self.protagonist_net.loss_f(priori_reward,ideal)
        loss.backward()
        self.protagonist_net.update()

        fl = float(loss)
        print(fl)



        # 主从网络同步
        if(self.sync_step>=2**7):
            self.assistant_net = copy.deepcopy(self.protagonist_net)
            self.sync_step=0
        else:
            self.sync_step+=1

        return 0