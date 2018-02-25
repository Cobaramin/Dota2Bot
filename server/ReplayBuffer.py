import random
from collections import deque

import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        batch = []
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            batch = random.sample(self.buffer, self.num_experiences)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        return states, actions, rewards, new_states, dones

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def add_multiple(self, data):
        state, action, reward, new_state, done = zip(*data)
        for i in range(len(state)):
            self.add(state[i], action[i], reward[i], new_state[i], done[i])

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def clear(self):
        self.buffer.clear()
        self.num_experiences = 0
