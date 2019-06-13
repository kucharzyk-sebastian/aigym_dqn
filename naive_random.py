import random
import gym
import tensorflow as tf

NUM_OF_EPISODES = 200
FRAMES_PER_EPISODE = 1000
BATCH_SIZE = 32
GAME_ID = 'CartPole-v1'
CPU_ID = '/device:CPU:0'


if __name__ == "__main__":
    with tf.device(CPU_ID):
        game = gym.make(GAME_ID)
        num_of_actions = game.action_space.n
        is_done = False
        for episode in range(NUM_OF_EPISODES):
            current_state = game.reset()
            for frame in range(FRAMES_PER_EPISODE):
                game.render()
                action = random.randrange(num_of_actions)
                new_state, gained_reward, is_done, info = game.step(action)
                if is_done:
                    print("episode: {0}/{1}; result: {2}".format(episode, NUM_OF_EPISODES, frame))
                    break
            if not is_done:
                print("episode: {0}/{1}; result: {2}".format(episode, NUM_OF_EPISODES, frame))
