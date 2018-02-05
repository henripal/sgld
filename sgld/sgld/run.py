import yaml
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

import sgld

parser = argparse.ArgumentParser()
parser.add_argument("paramset")

args = parser.parse_args()

try:
    with open('config.yaml', 'r') as f:
        params = yaml.load(f)
        try:
            parameters = params[args.paramset]
        except KeyError:
            print("Params {} not found in config.yaml file".format(args.paramset))
except OSError:
    print("config.yaml not  found.")

try:
    i = 0 
    while os.path.exists(args.paramset + '_' + str(i)):
        i += 1
    directory = args.paramset + '_' + str(i)
    os.makedirs(directory)
except OSError:
    print('unable to create directory')



if __name__ == '__main__':
    print("Using parameters: {}".format(parameters))

    with open(os.path.join(directory, 'config.yaml'), 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False)

    torch.cuda.set_device(parameters['cuda_device'])
    model = sgld.MnistModel()
    train_loader, test_loader = sgld.make_datasets()
    model = model.cuda()
    optimizer = sgld.SGLD(model.parameters(), lr=parameters['lr'])
    loss, acc, val, histo = sgld.train(
        model,
        parameters['epochs'],
        train_loader,
        test_loader,
        lambda x: sgld.lossrate(x, parameters['a'],
            parameters['b'], parameters['gamma']),
        False,
        optimizer
    )