

import gym


import matplotlib.pyplot as plt

import pickle

import dqn


class cart_pole():

    def reward_f(cart_pole_a,next_state):
        # 一个行为，会产生一个新状态，根据新状态来设计一个辅助的奖励函数（并非游戏实际奖励，所以这是用来指导主体行为的。）

        
        x, x_dot, theta, theta_dot = next_state

        r0 = 0.7
        r1 = abs(x)*(-1/2.4)
        r2 = (-1/0.2)*abs(theta)

        reward = r0+r1+r2

        # reward = 3-abs(x)

        return reward

    def main(cart_pole_a):
        
        env = gym.make('CartPole-v1')
        # env = gym.make('CartPole-v1',render_mode= 'human')
        env = env.unwrapped


        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]
        

        sp = [N_STATES,2,N_ACTIONS]
        # print(sp)


        lr = 0.01

        while(1):
            dqn_a = dqn.dqn(sp,lr,1)
            print('epsilon',dqn_a.epsilon)

            epi = 0
            stt = 0

            while(1):
                state,_ = env.reset()
                step=0
                while(1):
                    env.render()

                    action = dqn_a.action_f(state)


                    x,xd,th,thd = state
                    next_state, _, done, _, _ = env.step(action)
                    reward = cart_pole_a.reward_f(next_state)
    
                    packet = [state, action, next_state , reward]
                    dqn_a.store(packet)
                    # dqn_a.store(state, action, reward, next_state)
                    



                    state = next_state
                    step +=1
                    stt+=1
                    if done:
                        break
                
                epi+=1
                dqn_a.learn()



                print(epi,step)
                # break
                if(epi>=4000):
                    # break
                    pass

                if(epi>2**9):
                    if(step<30):
                        print('wrong,重开')
                        break

                print('____\n\n')


if __name__ == "__main__":
    cart_pole().main()