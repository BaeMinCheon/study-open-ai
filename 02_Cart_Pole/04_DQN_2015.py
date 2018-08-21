import numpy as np
import tensorflow as tf
import DQN
import random
import collections
import gym

environment = gym.make("CartPole-v0")
environment._max_episode_steps = 10001
inputSize = environment.observation_space.shape[0]  # 4
outputSize = environment.action_space.n             # 2
discount = 0.9
trainNumber = 1001

replayBuffer = collections.deque()
MAX_REPLAY_NUMBER = 50000

def GetStack(_md, _td, _tb):
    xStack = np.empty(0).reshape(0, _md.mInputSize)
    yStack = np.empty(0).reshape(0, _md.mOutputSize)

    for state, action, reward, nextState, isDone in _tb:
        Q = _md.Predict(state)

        if isDone:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + _md.mDiscount * _td.Predict(nextState)[0, np.argmax(_md.Predict(nextState))]

        xStack = np.vstack([xStack, state])
        yStack = np.vstack([yStack, Q])
    
    return xStack, yStack

with tf.Session() as sess:
    mainDQN = DQN.DQN(sess, inputSize, outputSize, discount)
    mainDQN.Initialize()
    targetDQN = DQN.DQN(sess, inputSize, outputSize, discount, "target")
    targetDQN.Initialize()
    sess.run(tf.global_variables_initializer())

    syncOps = mainDQN.GetSyncWeightOps("main", "target")
    sess.run(syncOps)

    stepSum = 0

    for i in range(trainNumber):
        boundary = 1.0 / ((i / 10) + 1)
        isDone = False
        stepCount = 0
        state = environment.reset()

        while not isDone:
            if np.random.rand(1) < boundary:
                action = environment.action_space.sample()
            else:
                action = np.argmax(mainDQN.Predict(state))
            
            nextState, reward, isDone, probability = environment.step(action)
            if isDone:
                reward = -100
            
            replayBuffer.append((state, action, reward, nextState, isDone))
            if len(replayBuffer) > MAX_REPLAY_NUMBER:
                replayBuffer.popleft()
            
            state = nextState
            stepCount += 1
            if stepCount > 10000:
                break

        stepSum += stepCount

        if i % 10 == 1:
            for j in range(50):
                batch = random.sample(replayBuffer, 10)
                xStack, yStack = GetStack(mainDQN, targetDQN, batch)
                error, train = mainDQN.Train(xStack, yStack)
            
            sess.run(syncOps)
            
            print("train number {} \t average step {}".format(i, stepSum / 10))
            stepSum = 0

    mainDQN.TestNetwork(environment)