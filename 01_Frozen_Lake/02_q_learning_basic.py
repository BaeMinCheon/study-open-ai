import gym
import numpy as np
import matplotlib.pyplot as plt
import random

def rargmax(vector):    # https://gist.github.com/stober/1943451
    """ Argmax that chooses randomly among eligible maximum idices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

gym.envs.registration.register(id='FrozenLake-v3',
entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4', 'is_slippery':False})
environment = gym.make('FrozenLake-v3')

Q = np.zeros([environment.observation_space.n, environment.action_space.n])
trainSize = 1000

rewardList = []
for i in range(trainSize):
    currState = environment.reset()
    totalReward = 0
    isDone = False

    while not isDone:
        action = rargmax(Q[currState, :])
        nextState, reward, isDone, probability = environment.step(action)
        Q[currState, action] = reward + np.max(Q[nextState, :])
        totalReward += reward
        currState = nextState

    rewardList.append(totalReward)

print("Success rate: " + str(sum(rewardList) / trainSize))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rewardList)), rewardList, color="blue")
plt.show()
