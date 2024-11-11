"""
Some quick utils to pull
from CIFAR-10 with minimal code-rewriting.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor

cifar10DataTrain = torchvision.datasets.CIFAR10('data/cifar10/', download=True, train=True, transform=ToTensor())
cifar10DataTest = torchvision.datasets.CIFAR10('data/cifar10/', download=True, train=False, transform=ToTensor())

cifarTrainset = lambda batchSize: torch.utils.data.DataLoader(cifar10DataTrain,
                                          batch_size=batchSize,
                                          shuffle=True)

cifarTestset = lambda batchSize: torch.utils.data.DataLoader(cifar10DataTest,
                                          batch_size=batchSize,
                                          shuffle=True)

