# 使用了神经网络的q_lean算法

import mlp
import random
import copy


class dqn(object):
    def __init__(self,sp,lr,bs):
        self.learn_rate = lr
        self.batch_size = bs

        self.capacity = 2**10   # 内存容量
        
        self.memory_pointer = 0 # 内存指针
        self.memory=list()
        for i in range(self.capacity):
            self.memory.append(list())

        self.protagonist_net  = mlp.mlp(sp,lr)
        self.assistant_net = copy.deepcopy(self.protagonist_net)

        self.sync_step = 0

    def action_f(self,state):

        return 0
    def store(self,packet):
    # def store(self,state, action, next_state, reward):
        # 存储一些经验
        # self.memory.append(packet)
        
        #　 指针循环存储数据
        smp = self.memory_pointer
        if(smp>=self.capacity):
            smp = 0
        else:
            smp+=1

        
        self.memory[smp]=packet
        self.memory_pointer = smp


        return 0
    def learn(self):

        # 随机取出一个批量
        ri = random.randint(0,self.capacity-self.batch_size)
        batch = self.memory[ri:ri+self.batch_size]

        beta = 0.9  # 即延系数
        alpha = 0.9 # 先后系数

        # 现在要把列表转成tensor 来批量运算。
        


        # total_reward = (1-beta) * instant_gratification + beta* deferred_gratification



        # 主从网络同步
        if(self.sync_step>=2**7):
            self.assistant_net = copy.deepcopy(self.protagonist_net)
            self.sync_step=0
        else:
            self.sync_step+=1

        return 0