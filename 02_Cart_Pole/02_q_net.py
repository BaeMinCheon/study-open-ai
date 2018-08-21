import gym
import numpy as np
import matplotlib as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

environment = gym.make('CartPole-v0')
learningRate = 0.1
inputSize = environment.observation_space.shape[0]  # 4
outputSize = environment.action_space.n             # 2

input = tf.placeholder(tf.float32, [None, inputSize])
weight = tf.get_variable("weight", shape=[inputSize, outputSize], initializer=tf.contrib.layers.xavier_initializer())
output = tf.matmul(input, weight)

label = tf.placeholder(shape=[1, outputSize], dtype=tf.float32)

error = tf.reduce_sum(tf.square(tf.subtract(label, output)))
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

discount = 0.99
trainNumber = 2001
totalStepList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(trainNumber):
        boundary = 1.0 / ((i / 10) + 1)
        stepCount = 0
        state = environment.reset()
        isDone = False

        while not isDone:
            stepCount += 1
            reshapedState = np.reshape(state, [1, inputSize])
            Q = sess.run(output, feed_dict={input: reshapedState})

            if np.random.rand(1) < boundary:
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q)

            nextState, reward, isDone, probability = environment.step(action)

            if isDone:
                Q[0, action] = -100
            else:
                reshapedNextState = np.reshape(nextState, [1, inputSize])
                Qmax = sess.run(output, feed_dict={input: reshapedNextState})
                Q[0, action] = reward + discount * np.max(Qmax)

            sess.run(train, feed_dict={input: reshapedState, label: Q})
            state = nextState

        totalStepList.append(stepCount)
        if i % 100 == 0:
            print(" train number {} \n total steps {}".format(i, stepCount))
            print()
        if len(totalStepList) > 10:
            if np.mean(totalStepList[-10 : ]) > 500:
                break

    print("TRAIN DONE")

    state = environment.reset()
    totalReward = 0
    while True:
        environment.render()
        reshapedState = np.reshape(state, [1, inputSize])
        Q = sess.run(output, feed_dict={input: reshapedState})
        action = np.argmax(Q)

        nextState, reward, isDone, probability = environment.step(action)
        totalReward += reward

        if isDone:
            print("total reward {}".format(totalReward))
            print("TEST DONE")
            break