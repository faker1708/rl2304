import gym
import numpy as np




env=gym.make('FrozenLake-v1',render_mode = 'human')
# env.reset()
# env.render()

state,_ = env.reset()
print("Action space: ", env.action_space)
print("Observation space: ", env.observation_space)




while(1):
    env.render()
    print(state)

    action = int(input())
    
    next_state, r, done, xxx, info = env.step(action)

    print(next_state, r, done, xxx, info)
    state = next_state
    print(done)


    if(done):
        break

