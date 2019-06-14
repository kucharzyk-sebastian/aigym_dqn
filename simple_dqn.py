import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True


class SimpleDqnNpc:
    def __init__(self, num_of_inputs, num_of_outputs):
        self._num_of_inputs = num_of_inputs
        self._num_of_outputs = num_of_outputs
        self._memory = deque(maxlen=1024)
        self._init_model()

    def _init_model(self):
        self._model = Sequential()
        self._model.add(Dense(16, input_dim=self._num_of_inputs, activation='linear'))
        self._model.add(Dense(16, activation='linear'))
        self._model.add(Dense(self._num_of_outputs, activation='linear'))
        self._model.compile(optimizer=SGD(), loss='mean_squared_error')

    def predict(self, state):
        act_profits = self._model.predict(state)
        return np.argmax(act_profits[0])

    def retain(self, current_state, taken_action, gained_reward, future_state, is_lost):
        self._memory.append((current_state, taken_action, gained_reward, future_state, is_lost))

    def replay(self, batch_size):
        batch = random.sample(self._memory, batch_size)
        for current_state, taken_action, gained_reward, future_state, is_lost in batch:
            future_act_best_profit = gained_reward
            if not is_lost:
                future_act_profits = self._model.predict(future_state)
                future_act_best_profit = np.amax(future_act_profits[0])
            current_act_profits = self._model.predict(current_state)
            current_act_profits[0][taken_action] = future_act_best_profit
            with tf.device('/device:GPU:0'):
                self._model.fit(x=current_state, y=current_act_profits, epochs=1, verbose=0)

    def load(self, model_path):
        self._model.load_weights(model_path)

    def save(self, model_path):
        self._model.save_weights(model_path)


NUM_OF_GAMES = 200
BATCH_SIZE = 32
if __name__ == "__main__":
    with tf.device('/device:CPU:0'):
        game = gym.make('CartPole-v1')
        observation_size = game.observation_space.shape[0]
        num_of_actions = game.action_space.n
        npc = SimpleDqnNpc(observation_size, num_of_actions)
        # agent.load("./save/cartpole-dqn.h5")
        is_done = False
        for e in range(NUM_OF_GAMES):
            current_state = game.reset()
            current_state = np.reshape(current_state, [1, observation_size])
            for frame in range(1000):
                game.render()
                action = npc.predict(current_state)
                new_state, gained_reward, is_done, info = game.step(action)
                gained_reward = gained_reward if not is_done else -1
                new_state = np.reshape(new_state, [1, observation_size])
                npc.retain(current_state, action, gained_reward, new_state, is_done)
                current_state = new_state
                if len(npc._memory) > BATCH_SIZE:
                    npc.replay(BATCH_SIZE)
                if is_done:
                    print("episode: {0}/{1}; result: {2}; used memory: {3}/{4}"
                          .format(e, NUM_OF_GAMES, frame, npc._memory.__len__(), npc._memory.maxlen))
                    break
            if not is_done:
                    print("episode: {0}/{1}; result: {2}; used memory: {3}/{4}"
                          .format(e, NUM_OF_GAMES, frame, npc._memory__len__(), npc._memory.maxlen))
            # if e % 10 == 0:
            # agent.save("./save/cartpole-dqn.h5")
