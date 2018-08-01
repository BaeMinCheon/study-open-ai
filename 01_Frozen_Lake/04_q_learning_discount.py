import gym
import numpy as np
import matplotlib.pyplot as plt

gym.envs.registration.register(id='FrozenLake-v3',
entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name':'4x4', 'is_slippery':False})
environment = gym.make('FrozenLake-v3')

Q = np.zeros([environment.observation_space.n, environment.action_space.n])
trainSize = 1000
discount = 0.9

rewardList = []
for i in range(trainSize):
    currState = environment.reset()
    totalReward = 0
    isDone = False

    while not isDone:
        action = np.argmax(
            Q[currState, :] + np.random.randn(1, environment.action_space.n) / (i + 1))
        nextState, reward, isDone, probability = environment.step(action)
        Q[currState, action] = reward + discount * np.max(Q[nextState, :])
        totalReward += reward
        currState = nextState

    rewardList.append(totalReward)

print("Success rate: " + str(sum(rewardList) / trainSize))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rewardList)), rewardList, color="blue")
plt.show()
