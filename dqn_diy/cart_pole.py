

import gym


import matplotlib.pyplot as plt

import pickle

import dqn


class cart_pole():

    def reward_f(cart_pole_a,next_state):
        return 0

    def main(cart_pole_a):
        
        env = gym.make('CartPole-v1')
        env = gym.make('CartPole-v1',render_mode= 'human')
        env = env.unwrapped


        N_ACTIONS = env.action_space.n
        N_STATES = env.observation_space.shape[0]

        sp = [N_STATES,50,N_ACTIONS]
        # print(sp)


        lr = 0.03
        dqn_a = dqn.dqn(sp,lr)

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
   
                packet = [state, action, reward, next_state ]
                dqn_a.store(packet)
                # dqn_a.store(state, action, reward, next_state)
                



                state = next_state
                step +=1
                stt+=1

                
                print(step,state,next_state)

                if done:
                    break

            epi+=1
            dqn.learn()

            break


if __name__ == "__main__":
    cart_pole().main()