

cart_pole实验

功能

让小车稳定在初始位置附近

原理：用pid算法改进了奖励函数


复现操作：
1   将train 的dqn_init 改成1
2   将train 的 保存模型阈值改成1

跑几轮生成一个模型 命名为a

3   将train 的dqn_init 改成0，读取a模型
4   将train 的 保存模型阈值改成0.24 或者更小

跑几轮生成一个模型 命名为b

5   load b 这个模型



不足之处：位置是稳定了，但游戏失败率增加了。之前的模型基本没见过失败的。现在的模型往往会失败。
但好处是一旦稳定运行，就可以宣布游戏不会结束了，杆子不会倒下，小车也不会偏太远。







