import numpy as np

class MovingAverage:
    def __init__(self, window_size):
        self.i = 0
        self.window_size = window_size

        self.reset()

    def add(self, n):
        self.window[self.i] = n
        self.i = (self.i + 1) % self.window_size
        self.valid_size = self.valid_size + 1 if self.valid_size < self.window_size else self.valid_size
        return self

    def __add__(self, n):
        return self.add(n)

    def get(self):
        if self.valid_size == 0:
            return 0.0
        return self.window[0:self.valid_size].sum() / self.window[0:self.valid_size].size

    def __float__(self):
        return self.get()
    
    def reset(self):
        self.window = np.zeros(self.window_size)
        self.valid_size = 0