

手写dqn

它存在一个即时满足与延时满足的问题，
还存在一个先验知识与后验知识的问题

所以应该是有两个系数来分别控制。


ideal = (1-alpha)*previous + alpha* after
after = (1-beta)* instance + beta*delay

这个后验就对应于 实际值 
先验对应于 估计值 

对于神经网络模型，也就是dqn来说，alpha 可加可不加吧。

而beta,则是用估计值来代替实际值 的一种近似方法。


到时候可以试试看有没有区别。

