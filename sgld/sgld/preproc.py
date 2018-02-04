import numpy as np
import PIL

class NoiseTransform:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, mnist_datapoint):
        img = np.asarray(mnist_datapoint)

        img = self.normalize(img)
        img = self.addnoise(img)
        img = self.denormalize(img)

        return PIL.Image.fromarray(img)

    def normalize(self, arr):
        mx = np.max(arr)
        arr = arr/mx
        return arr - np.mean(arr)

    def denormalize(self, arr):
        mx = np.max(arr)
        mn = np.min(arr)
        arr = (arr - mn)/(mx - mn)
        arr = arr * 255
        return np.uint8(arr)

    def addnoise(self, arr):
        noise = np.random.randn(arr.shape[0]*arr.shape[1]).reshape(arr.shape)
        noise = self.sigma * noise
        return arr + noise