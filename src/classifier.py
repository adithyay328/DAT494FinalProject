"""
This module defines the RESNET-50 classifier, as well
as logic to load it automatically as a module
"""
MODEL_DUMP_NAME = "res_class.model"

import os

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchvision.transforms import v2

from cifar import cifarTrainset

# Set all random seeds
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix which weights were chosen
chosenWeights = ResNet50_Weights.IMAGENET1K_V2
resnet = resnet50(weights=chosenWeights).eval().to(device)
resPreproc = chosenWeights.transforms()

train_nodes, eval_nodes = get_graph_node_names(resnet)
# Which layers to pull features from
layersToPull = {
  'layer4.2.add' : 'out'
}

featureExtractor = create_feature_extractor(resnet, return_nodes=layersToPull)

class RESNETClassifier(nn.Module):
  def __init__(self):
    # The classifier won't actually contain
    # resnet, we'll just use the above module
    # + feature extractor, and then a sequential
    # model afterwards to map to a 10 class output
    super().__init__()

    # The feature map size is 2048, 7, 7, so just go
    # from there
    self.model = nn.Sequential(
      nn.Conv2d( 2048, 512, 3 ),
      nn.BatchNorm2d( 512),
      nn.ReLU(),
      nn.Conv2d( 512, 64, 3 ),
      nn.BatchNorm2d( 64),
      nn.ReLU(),
      nn.Conv2d( 64, 32, 3),
      nn.BatchNorm2d( 32 ),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear( 32, 10 ),
      nn.Softmax()
    )

  def forward(self, X):
    with torch.no_grad():
      X = featureExtractor( resPreproc( X ) )["out"]

    return self.model ( X )

c = RESNETClassifier().to(device)

# If model file is in the directory,
# load weights into c
if os.path.exists( MODEL_DUMP_NAME ):
  c.load_state_dict( torch.load( MODEL_DUMP_NAME ) )

trainLoad = cifarTrainset(batchSize=128)
validLoad = cifarTrainset(batchSize=128)

# First, let's just overfit on one batch
# NOTES FROM ADI: We're getting stuck
# at a high loss, and a low accuracy.

# What could be wrong?
# 1. Check for vanishing gradients.
# RESULT: Yeah, it's a vanishing gradient. All grad
# norms are going to 0. Time to fix it.

# UPDATE 1: Dropping lr to 1e-4 lets us train for longer, and
# we get to 90% train accuracy before hitting another vanishing
# point. Lemme try BNs, and then increasing LR

# UPDATE 2: After adding batch norms, I can run a ludicrous
# LR of 1e-1, and gets to 100% accuracy really fast. Going
# to move to full training soon.

batch = next(iter(trainLoad))

print(f"Batch Y : {batch[1]}")

optim = torch.optim.Adam( c.parameters(), lr=1e-3 )
lossFn = nn.CrossEntropyLoss()
epochs = 10

# Epoch level train
# and valid accuracies
trainAccs = []
vaidAccs = []

for epoch in range(epochs):
  epochTotalTrainCorrects = 0
  epochTotalTrainSamples = 0

  for X, y in iter ( trainLoad ):
    X = X.to(device)
    y = y.to(device)

    modPred = c (  X  )
    optim.zero_grad()

    loss = lossFn( modPred, y)
    loss.backward()

    optim.step()

    epochTotalTrainCorrects += torch.sum( torch.argmax( modPred, dim=-1) == y).item()
    epochTotalTrainSamples += y.shape[0]

  epochTrainAcc = epochTotalTrainCorrects / epochTotalTrainSamples
  trainAccs.append( epochTrainAcc )
  print(f"Epoch {epoch} Train Acc: {epochTrainAcc}")

  # Now, do validation
  with torch.no_grad():
    epochTotalValidCorrects = 0
    epochTotalValidSamples = 0
    for X, y in iter( validLoad ):
      X = X.to(device)
      y = y.to(device)
      modPred = c( X )

      epochTotalValidCorrects += torch.sum( torch.argmax( modPred, dim=-1) == y).item()
      epochTotalValidSamples += y.shape[0]

    epochValidAcc = epochTotalValidCorrects / epochTotalValidSamples
    vaidAccs.append( epochValidAcc )

    print(f"Epoch {epoch} Valid Acc: {epochValidAcc}")

# Save model to disk
with open( MODEL_DUMP_NAME, "wb") as f:
  torch.save( c.state_dict(), f)
