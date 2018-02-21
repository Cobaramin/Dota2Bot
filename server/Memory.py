import numpy as np
from config import BATCH_SIZE, BUFFER_SIZE

class Memory:
    def __init__(self):
        self.curr_idx = 0
        self.mem_size = BUFFER_SIZE
        self.mem = np.array([None for _ in range(self.mem_size)], dtype=np.object)
        self.priority = np.array([0 for _ in range(self.mem_size*2-1)], dtype=np.float32)
        self.priority_max = 1000
        self.batch_size = BATCH_SIZE
        self.batch_idx = np.zeros(self.batch_size, dtype=np.int32)

    def full(self):
        return self.mem[self.curr_idx] != None

    def insert(self, data):
        self.mem[self.curr_idx] = data

        leaf = self.mem_size - 1 + self.curr_idx
        self.priority[leaf] = self.priority_max
        self.update_priority(leaf)

        self.curr_idx = (self.curr_idx + 1) % self.mem_size

    def update_priority(self, leaf):
        parent = (leaf + 1) // 2 - 1
        while parent >= 0:
            child_right = (parent + 1) * 2
            child_left = child_right - 1

            self.priority[parent] = self.priority[child_left] + self.priority[child_right]

            parent = (parent + 1) // 2 - 1

    def get_batch(self):
        batch_priority = np.random.rand(self.batch_size) * self.priority[0]
        for i in range(self.batch_size):
            priority = batch_priority[i]
            parent = 0
            while parent < self.mem_size - 1:
                child_right = (parent + 1) * 2
                child_left = child_right - 1

                if priority <= self.priority[child_left]:
                    parent = child_left
                else:
                    parent = child_right
                    priority -= self.priority[child_left]

            self.batch_idx[i] = parent - self.mem_size + 1

        return self.mem[self.batch_idx]

    def update_batch_td_error(self, td_error):
        priority = np.abs(td_error)
        for i in range(self.batch_size):
            leaf = self.mem_size - 1 + self.batch_idx[i]
            self.priority[leaf] = priority[i]
            self.update_priority(leaf)
