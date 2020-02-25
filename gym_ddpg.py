import filter_env
from ddpg import *
import gc
gc.enable()

ENV_NAME = 'Pendulum-v0'
EPISODES = 1000
TEST = 10

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    # env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in range(env.spec.max_episode_steps):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.max_episode_steps):
                    #env.render()
                    #direct action for test
                    action = agent.action(state)
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode: {}; Evaluation Average Reward: {}'.format(episode,ave_reward))
    # env.monitor.close()

if __name__ == '__main__':
    main()
