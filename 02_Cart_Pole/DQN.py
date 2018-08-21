import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, _ss, _is, _os, _dc, _nn="main"):
        self.mSession = _ss
        self.mInputSize = _is
        self.mOutputSize = _os
        self.mDiscount = _dc
        self.mNetworkName = _nn

    def Initialize(self, _hs=10, _lr=1e-1):
        with tf.variable_scope(self.mNetworkName):
            self.mInput = tf.placeholder(tf.float32, [None, self.mInputSize], name="input")
            
            weight01 = tf.get_variable("weight01", shape=[self.mInputSize, _hs], initializer=tf.contrib.layers.xavier_initializer())
            output01 = tf.nn.tanh(tf.matmul(self.mInput, weight01))

            weight02 = tf.get_variable("weight02", shape=[_hs, self.mOutputSize], initializer=tf.contrib.layers.xavier_initializer())
            output02 = tf.matmul(output01, weight02)

            self.mOutput = output02

        self.mLabel = tf.placeholder(shape=[None, self.mOutputSize], dtype=tf.float32)
        self.mError = tf.reduce_mean(tf.square(tf.subtract(self.mLabel, self.mOutput)))
        self.mTrain = tf.train.AdamOptimizer(learning_rate=_lr).minimize(self.mError) 

    def Predict(self, _st):
        reshapedState = np.reshape(_st, [1, self.mInputSize])
        return self.mSession.run(self.mOutput, feed_dict={self.mInput: reshapedState})

    def Train(self, _xs, _ys):
        return self.mSession.run([self.mError, self.mTrain], feed_dict={self.mInput: _xs, self.mLabel: _ys})

    def TestNetwork(self, _ev):
        state = _ev.reset()
        totalReward = 0

        while True:
            _ev.render()
            action = np.argmax(self.Predict(state))
            state, reward, isDone, probability = _ev.step(action)
            totalReward += reward

            if isDone:
                print("total reward : {}".format(totalReward))
                print()
                break

    @staticmethod
    def GetSyncWeightOps(_sn="main", _dn="target"):
        srcVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=_sn)
        dstVarList = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=_dn)

        opList = []
        for srcVar, dstVar in zip(srcVarList, dstVarList):
            opList.append(dstVar.assign(srcVar.value()))
        
        return opList


print()
if __name__ == "__main__":
    print("\t DQN.py says : I'm running as main")
else:
    print("\t DQN.py says : I'm running as import")
print()