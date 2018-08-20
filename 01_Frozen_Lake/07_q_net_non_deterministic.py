import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def GetOneHot(x):
    return np.identity(16)[x : x + 1]

environment = gym.make("FrozenLake-v0")
inputSize = environment.observation_space.n     # 16
outputSize = environment.action_space.n         # 4
learningRate = 0.1

input = tf.placeholder(shape=[1, inputSize], dtype=tf.float32)
weight = tf.Variable(tf.random_uniform([inputSize, outputSize], minval=0.0, maxval=0.01))
output = tf.matmul(input, weight)

label = tf.placeholder(shape=[1, outputSize], dtype=tf.float32)

error = tf.reduce_sum(tf.square(tf.subtract(label, output)))
train = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(error)

discount = 0.99
trainNumber = 2001
rewardList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(trainNumber):
        state = environment.reset()
        boundary = 1.0 / ((i / 50) + 10)
        totalReward = 0
        isDone = False

        while not isDone:
            Q = sess.run(output, feed_dict={input: GetOneHot(state)})

            if np.random.rand(1) < boundary:
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q)

            nextState, reward, isDone, probability = environment.step(action)

            if isDone:
                Q[0, action] = reward
            else:
                Qmax = sess.run(output, feed_dict={input: GetOneHot(nextState)})
                Q[0, action] = reward + discount * np.max(Qmax)

            sess.run(train, feed_dict={input: GetOneHot(state), label: Q})
            totalReward += reward
            state = nextState

        rewardList.append(totalReward)

        if i % 100 == 0:
            print("train number {}".format(i))

print("success rate {}".format(sum(rewardList) / trainNumber))
plt.bar(range(len(rewardList)), rewardList, color='b', alpha=0.4)
plt.show()

print("\n \t TRAIN DONE")