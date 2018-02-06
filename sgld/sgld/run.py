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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paramset")
    parser.add_argument("cuda_device", type=int)

    args = parser.parse_args()

    parameters, directory = sgld.get_params(args.paramset)
    cuda_device = args.cuda_device
    print("Using parameters: {}".format(parameters))

    with open(os.path.join(directory, 'config.yaml'), 'w') as f:
        yaml.dump(parameters, f, default_flow_style=False)

    loss, acc, val, histo, modelparams = sgld.runall(
        cuda_device, parameters['lr'],
        parameters['epochs'], parameters['a'], parameters['b'],
        parameters['gamma'], parameters['addnoise']) 
    

    np.save(os.path.join(directory, 'loss.npy'), loss)
    np.save(os.path.join(directory, 'acc.npy'), acc)
    np.save(os.path.join(directory, 'val.npy'), val)
    np.save(os.path.join(directory, 'histo.npy'), histo)
    torch.save(modelparams, os.path.join(directory, 'model.tch'))


if __name__ == '__main__':
    main()