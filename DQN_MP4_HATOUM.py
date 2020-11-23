import gym
import random
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

env = gym.make('LunarLander-v2')

class DQAgent:

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = 1e-2
        self.lr = 1e-3
        self.epsilon_decay = .996
        self.memory = deque()
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(128, input_dim=self.state_space, activation=relu)) #entrée : state_space
        model.add(Dense(128, activation=relu))
        model.add(Dense(self.action_space, activation=linear)) #sortie action_space
        model.compile(loss='mse', optimizer=adam())
        return model

    #epsilon greedy policy
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space) #action aléatoire
        act_values = self.model.predict(state) #action avec plus grande récompense
        return np.argmax(act_values[0])


    def learn(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        index = np.array([i for i in range(self.batch_size)])
        targets_full[[index], [actions]] = targets


        self.model.fit(states, targets_full, epochs=1, verbose=0) #apprentissage du réseau, mise à jour des poids

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_DQAgent(nb_episode):

    rewards = []
    agent = DQAgent(env.action_space.n, env.observation_space.shape[0])
    for e in range(nb_episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.choose_action(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            agent.learn()
            if done:
                print("episode: {}/{}, reward: {}".format(e, nb_episode, score))
                break
        rewards.append(score)

    return rewards


if __name__ == '__main__':
    rewards = train_DQAgent(400) #entrainement avec 500 épisodes
    plt.plot([i+1 for i in range(0, len(rewards), 2)], rewards[::2])
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()
