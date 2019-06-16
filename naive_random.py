import random
import gym

NUM_OF_AGENTS = 4
NUM_OF_EPISODES = 100
FRAMES_PER_EPISODE = 1000
GAME_ID = "LunarLander-v2"

if __name__ == "__main__":
    game = gym.make(GAME_ID)
    num_of_actions = game.action_space.n
    is_done = False
    avgs = []
    for model in range(NUM_OF_AGENTS):
        scores = []
        for episode in range(NUM_OF_EPISODES):
            score = 0
            current_state = game.reset()
            for frame in range(FRAMES_PER_EPISODE):
                # game.render()
                action = random.randrange(num_of_actions)
                new_state, gained_reward, is_done, info = game.step(action)
                score += gained_reward
                if is_done:
                    break
                scores.append(score)
        avgs.append(sum(scores)/len(scores))
    game.close()
    for i, avg in enumerate(avgs):
        print("Model {} has avarage: {}".format(i, avg))
    print("Overall avg: {}".format(sum(avgs)/len(avgs)))

