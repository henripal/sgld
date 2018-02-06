import numpy as np

def lossrate(t, a, b, gamma):
    return a * np.power(b, gamma) / np.power((t + b), gamma)
