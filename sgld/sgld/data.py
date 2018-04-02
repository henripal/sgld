import torch.utils
from torchvision import datasets, transforms
from .preproc import NoiseTransform

def make_datasets(bs=1024, test_bs=4096, noise=0):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
            train=True,
            download=True,
            transform=transforms.Compose([NoiseTransform(noise),
                                     transforms.ToTensor()])),
        batch_size=bs,
        shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data',
            train=False,
            transform=transforms.Compose([NoiseTransform(noise),
                                        transforms.ToTensor()])),
        batch_size=test_bs)

    return train_loader, test_loader