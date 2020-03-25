import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

EPISODES = 201

class AdvancedDqnNpc:
    def __init__(self, num_of_inputs, num_of_outputs):
        self._num_of_inputs = num_of_inputs
        self._num_of_outputs = num_of_outputs
        self._memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self._exploration_rate = 1.0  # exploration rate
        self._exploration_rate_min = 0.1
        self._exploration_rate_decay = 0.995
        self.learning_rate = 0.001
        self._model = self._init_model()

    def _init_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(150, input_dim=self._num_of_inputs, activation='relu'))
        model.add(Dense(120, activation='sigmoid'))
        model.add(Dense(self._num_of_outputs, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def retain(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        if np.random.rand() <= self._exploration_rate:
            return random.randrange(self._num_of_outputs)
        act_values = self._model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self._memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                pred = self._model.predict(next_state)
                amax = np.amax(pred[0])
                target = (reward + self.gamma * amax)
            target_f = self._model.predict(state)
            target_f[0][action] = target
            with tf.device('/device:GPU:0'):
                self._model.fit(x=state, y=target_f, epochs=1, verbose=0)
        if self._exploration_rate > self._exploration_rate_min:
            self._exploration_rate *= self._exploration_rate_decay

    def load(self, name):
        self._model.load_weights(name)

    def save(self, name):
        self._model.save_weights(name)


if __name__ == "__main__":
    with tf.device('/device:CPU:0'):
        env = gym.make('LunarLander-v2')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = AdvancedDqnNpc(state_size, action_size)
        # agent.load("advanced1-dqn.h5")
        done = False
        batch_size = 16

        for e in range(EPISODES):
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(1000):
                env.render()
                action = agent.predict(state)
                next_state, reward, done, info = env.step(action)
                score += reward
                if reward < 0:
                    reward *= 100
                next_state = np.reshape(next_state, [1, state_size])
                agent.retain(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}, time {}"
                          .format(e, EPISODES, score, agent._exploration_rate, time))
                    break
                if len(agent._memory) > batch_size:
                   agent.replay(batch_size)
            if not done:
                    print(reward)
                    print("episode: {}/{}, score: {}, e: {:.2}, time {}"
                          .format(e, EPISODES, score, agent._exploration_rate, time))
            #if e % 10 == 0:
                #agent.save("advanced-dqn.h5")
