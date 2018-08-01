import gym
import numpy as np
import matplotlib.pyplot as plt

environment = gym.make('FrozenLake-v0')

Q = np.zeros([environment.observation_space.n, environment.action_space.n])
trainSize = 1000
discount = 0.99
learningRate = 0.85

rewardList = []
for i in range(trainSize):
    currState = environment.reset()
    totalReward = 0
    isDone = False

    while not isDone:
        action = np.argmax(
            Q[currState, :] + np.random.randn(1, environment.action_space.n) / (i + 1))
        nextState, reward, isDone, probability = environment.step(action)
        Q[currState, action] = (1 - learningRate) * Q[currState, action] + learningRate * (
            reward + discount * np.max(Q[nextState, :]))
        totalReward += reward
        currState = nextState

    rewardList.append(totalReward)

print("Success rate: " + str(sum(rewardList) / trainSize))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rewardList)), rewardList, color="blue")
plt.show()
