import random


class ReplayBuffer:
    """A simple buffer to save previous samples, in order to mix these into batches when doing incremental training.
    Has a maximum capacity in order to maintain memory footprint. If full it overwrites memory evenly distributed over
    the buffer to ensure even weight to older and newer batches in the buffer.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []  # the buffer is a list, add (batch_x, batch_y) to it

    def add(self, data):
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            idx = random.randint(0, len(self.buffer) - 1)
            self.buffer[idx] = data

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
