import tensorflow
from tensorflow.python.keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.python.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import huskarl as hk
# from huskarl.policy import Greedy, GaussianEpsGreedy

import gym
import gym_foo
# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

create_env = lambda: gym.make('foo-v0').unwrapped
env = create_env()

score_requirement = 50
count = 0


# Build a simple neural network with 3 fully connected layers as our model
model = Sequential([
    Dense(20, activation='relu', input_shape=env.observation_space.shape),
    Dense(16, activation='relu'),
    Dense(10, activation='relu'),
])

# Create Deep Q-Learning Network agent
# agent = hk.agent.DQN(model, actions=env.action_space.n, nsteps=2)

# We will be running multiple concurrent environment instances
instances = 4
# Create a policy for each instance with a different distribution for epsilon
policy = [hk.policy.Greedy()] + [hk.policy.GaussianEpsGreedy(eps, 0.1) for eps in np.arange(0, 1, 1 / (instances - 1))]
# Create Advantage Actor-Critic agent
agent = hk.agent.A2C(model, actions=env.action_space.n, nsteps=5, instances=instances, policy=policy)


def plot_rewardsA2C(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for i, (ed, steps) in enumerate(zip(episode_rewards, episode_steps)):
        plt.plot(steps, ed, alpha=0.5 if i == 0 else 0.2, linewidth=2 if i == 0 else 1)
    plt.show() if done else plt.pause(0.001)  # Pause a bit so that the graph is updated


# Create simulation, train and then test
sim = hk.Simulation(create_env, agent)
sim.train(max_steps=6000, instances=instances, plot=plot_rewardsA2C)
sim.test(max_steps=2000)

