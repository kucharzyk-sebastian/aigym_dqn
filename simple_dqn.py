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
    "Klasa implementująca agenta DQN opartego o prostą sieć neuronową"

    def __init__(self, num_of_inputs, num_of_outputs):
        """
        num_of_inputs - długość wektora będącego wejściem dla sieci neuronowej
        num_of_outputs - ilość wyjść z sieci neuronowej
        """

        self._num_of_inputs = num_of_inputs
        self._num_of_outputs = num_of_outputs
        self._exploration_rate = 1.0  # exploration rate
        self._exploration_rate_min = 0.1
        self._exploration_rate_decay = 0.997
        self.memory = deque(maxlen=1024)
        self._init_model()

    def _init_model(self):
        """
        Inicjalizuje model sieci neuronowej.
        Wybraliśmy (w naszym mniemaniu) najproszte parametry i kształt.
        """

        self._model = Sequential()
        self._model.add(Dense(self._num_of_inputs, input_dim=self._num_of_inputs, activation='linear'))
        self._model.add(Dense(self._num_of_outputs, activation='linear'))
        self._model.compile(optimizer=SGD(), loss='mean_squared_error')

    def act(self, state):
        """Przewiduje i zwraca akcję, którą należy wykonać"""

        if np.random.rand() <= self._exploration_rate:
            return random.randrange(self._num_of_outputs)
        act_values = self._model.predict(state)
        return np.argmax(act_values[0])

    def retain(self, current_state, taken_action, gained_reward, next_state, is_done):
        """Zapisuje dyn przypadku w pamięci agenta"""

        self.memory.append((current_state, taken_action, gained_reward, next_state, is_done))

    def replay(self, batch_size):
        """
        Doszkala sieć neuronową na losowym fragmencie z jego pamięci
        batch-size - rozmiar fragmentu pamięci
        """

        batch = random.sample(self.memory, batch_size)
        for current_state, taken_action, gained_reward, next_state, is_done in batch:
            next_act_best_profit = gained_reward
            if not is_done:
                future_act_profits = self._model.predict(next_state)
                next_act_best_profit = np.amax(future_act_profits[0])
            current_act_profits = self._model.predict(current_state)
            current_act_profits[0][taken_action] = next_act_best_profit
            with tf.device('/device:GPU:0'):
                self._model.fit(x=current_state, y=current_act_profits, epochs=1, verbose=0)
        if self._exploration_rate > self._exploration_rate_min:
            self._exploration_rate *= self._exploration_rate_decay
        else:
            self._exploration_rate = 0.0

    def load(self, model_path):
        """Wczytuje model z pamięci"""

        self._model.load_weights(model_path)

    def save(self, model_path):
        """Zapisuje modele do pamięci"""

        self._model.save_weights(model_path)


NUM_OF_AGENTS = 1
NUM_OF_EPISODES = 50
FRAMES_PER_EPISODE = 1000
BATCH_SIZE = 16
GAME_ID = "LunarLander-v2"

if __name__ == "__main__":
    with tf.device('/device:CPU:0'):
        game = gym.make(GAME_ID)
        num_of_actions = game.action_space.n
        observation_size = game.observation_space.shape[0]
        npc = SimpleDqnNpc(observation_size, num_of_actions)
        is_done = False
        avgs = []
        for model in range(NUM_OF_AGENTS):
            scores = []
            for episode in range(NUM_OF_EPISODES):
                score = 0
                current_state = np.reshape(game.reset(), [1, observation_size])
                for frame in range(FRAMES_PER_EPISODE):
                    # game.render()
                    action = npc.act(current_state)
                    new_state, gained_reward, is_done, info = game.step(action)
                    new_state = np.reshape(new_state, [1, observation_size])
                    npc.retain(current_state, action, gained_reward, new_state, is_done)
                    score += gained_reward
                    current_state = new_state
                    if len(npc.memory) > BATCH_SIZE:
                        npc.replay(BATCH_SIZE)
                    if is_done:
                        print("episode: {0}/{1}; result: {2}; e: {3} used memory: {4}/{5}"
                              .format(episode, NUM_OF_EPISODES, score, npc._exploration_rate, len(npc.memory), npc.memory.maxlen))
                        break
                scores.append(score)
                if not is_done:
                    print("episode: {0}/{1}; result: {2}; used memory: {3}/{4}"
                          .format(episode, NUM_OF_EPISODES, score, len(npc.memory), npc.memory.maxlen))
            npc.save("simple_dqn_" + str(model) + ".h5")
            avgs.append(sum(scores) / len(scores))
        for i, avg in enumerate(avgs):
            print("Model {} has avarage: {}".format(i, avg))
        print("Overall avg: {}".format(sum(avgs) / len(avgs)))
