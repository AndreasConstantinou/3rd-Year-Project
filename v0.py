import tensorflow
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt

import huskarl as hk
import gym
import gym_foo
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# from gym_foo.envs.FooEnv import FooEnv
# tmp = FooEnv()
#
# print(tmp.render())
create_env = lambda: gym.make('foo-v0').unwrapped
env = create_env()


score_requirement = 50
count=0

def randomAggent ():
    reward = 0
    done = False
    env.reset()
    while not (reward == score_requirement):
        env.render()
        env.step(env.action_space.sample())  # take a random action

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("done")
            print(env.state)
            break
    env.close()
    print(env.state)


# Build a simple neural network with 3 fully connected layers as our model
model = Sequential([
  Dense(16, activation='relu', input_shape=env.observation_space.shape),
  Dense(16, activation='relu'),
  Dense(16, activation='relu'),
])
# Create Deep Q-Learning Network agent
agent = hk.agent.DQN(model, actions=env.action_space.n, nsteps=1)


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
        plt.pause(0.001)
        plt.draw()

        # plt.draw()# Pause a bit so that the graph is updated


# Create simulation, train and then test
sim = hk.Simulation(create_env, agent)
sim.train(max_steps=5000, visualize=True,plot=plot_rewards)
sim.test(max_steps=1000)



# randomAggent()


