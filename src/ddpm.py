"""
A small implementation of DDPM on CIFAR-10.

A lot of the scheduling hints and and hyper-paramaters
are from the DDPM paper.

To make this more fun, I'm going to use a VisionTransformer
as in Diffusion Transformers by Peebles
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

from cifar import *

class CIFARProcess:
  """
  A class that provides
  utilities and testing logic
  to make sure the diffusion process
  works properly for this dataset.
  It's really easy to screw this up,
  so this should provide all the functionality we need
  """
  def __init__(self, minBeta = 1e-4, maxBeta = 1e-2, steps = 1000):
    self.minBeta = minBeta
    self.maxBeta = maxBeta
    self.steps = steps

    self.allBetas = torch.tensor(np.linspace ( minBeta, maxBeta, num=steps ))
    self.alphas = 1 - self.allBetas
    self.alphaBars = torch.cumprod(self.alphas, dim=0)

  def noiseForward( self, X, noise = None, timesteps = None ):
    """
    Given an X from the dataset, and optionally noise and timesteps,
    generates all the data we need for training the model for reversing
    """
    if timesteps is None:
      timesteps = torch.randint(1, self.steps, (X.shape[0],))

    timestepsOneDown = timesteps - 1 # Use this to get the mean for the prior step.

    selectedAlphaBars = self.alphaBars[timesteps].squeeze()
    alphaBarsOneDown = self.alphaBars[timestepsOneDown].squeeze()

    if noise is None:
      noise = torch.randn_like(X)
    
    sqrtAlphaBars = torch.sqrt(selectedAlphaBars).to(X)
    sqrtAlphaBarsOneDown = torch.sqrt(alphaBarsOneDown).to(X)
    while len ( sqrtAlphaBars.shape ) < len ( X.shape ):
      sqrtAlphaBars = sqrtAlphaBars.unsqueeze(-1)
      sqrtAlphaBarsOneDown = sqrtAlphaBarsOneDown.unsqueeze(-1)

    noiseX =  X * sqrtAlphaBars.expand_as ( X ) + (1-sqrtAlphaBars) * noise
    XMeanOneDown = X * sqrtAlphaBarsOneDown.expand_as ( X )

    noiseX = noiseX.to(X)
    XMeanOneDown = XMeanOneDown.to(X)
    noise = noise.to(X)

    return {
      "noiseX" : noiseX,
      "XMeanOneDown" : XMeanOneDown,
      "noise" : noise,
      "timesteps" : timesteps
    }
  
  def reverseMean( self, noisyX, meanPred, destinationTimesteps, noise = None ):
    """
    Given the noisy inputX, and the model's predicted posterior mean,
    and the destination timestep, reverse the process.
    """
    sqrtAlphas = torch.sqrt(self.allBetas[destinationTimesteps])
    return meanPred + torch.randn_like(meanPred) * sqrtAlphas.to(meanPred)

  def visualizeForward(self, X):
    """
    Given a single sample, visualize the forward process,
    as a GIF. Just helps to ensure that the noising is
    working properly. Also, print the mean and variance
    at the last timestep
    """
    imgs = []
    for it in range(self.steps):
      noiseDict = self.noiseForward ( X, timesteps = torch.tensor([it]).to(X) )
      imgs.append((noiseDict["noiseX"].numpy(force=True).transpose(1,2,0)*255).astype(np.uint8))

    # Make a gif out of it
    imageio.mimsave("forward.gif", imgs)

    # Also, print final mean and var
    print ( f"Final step mean and var: {noiseDict['noiseX'].mean()}, {noiseDict['noiseX'].var()}" )

  def visualizeReverse(self, X):
    """
    Given a single sample, visualizes the reverse process.

    To do this, we begin by picking a sample at the highest noise level.
    Then, we use the reverse mean function to get samples from the past, BUT
    using a mean function that knows the exact mean for this sample
    by directly computing it. This should verify that a good mean function
    can go back properly
    """
    noise = torch.randn_like(X)
    imgs = [ (noise.numpy(force=True).transpose(1,2,0)*255).astype(np.uint8) ]
    for it in range(self.steps-1, -1, -1):
      perfectMean = self.noiseForward ( X, noise = noise, timesteps = torch.tensor([it]).to(X) )["XMeanOneDown"]
      noise = self.reverseMean ( noise, perfectMean, it )

      imgs.append((noise.numpy(force=True).transpose(1,2,0)*255).astype(np.uint8))

    imageio.mimsave("reverse.gif", imgs)

class ResdiualMLPLayer( nn.Module ):
  def __init__(self, nFeatures ):
    super(ResdiualMLPLayer, self).__init__()
    self.li = nn.Linear(nFeatures, nFeatures)
  
  def forward(self, X):
    return nn.functional.relu ( nn.functional.relu ( self.li ( X ) ) + X )

class ResidualCNNLayer( nn.Module ):
  def __init__(self, nFeatures ):
    super(ResidualCNNLayer, self).__init__()
    self.li = nn.Conv2d(nFeatures, nFeatures, 3, padding=1)

  def forward(self, X):
    return nn.functional.relu ( nn.functional.relu ( self.li ( X ) ) + X )


# To keep models simple,
# let's just use a fucking MLP
class CIFARMLP(nn.Module):
  def __init__(self):
    super(CIFARMLP, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d ( 3, 32, 3, padding=1),
      nn.ReLU(),
     *[ ResidualCNNLayer(32) ]*4,
      nn.Flatten(),
      nn.Linear(int(3*32*32*2*32/3), 512),
      nn.ReLU(),
    *[ResdiualMLPLayer(512)]*4,
      nn.Linear(512, 3*32*32),
      nn.Sigmoid()
    )

  def forward(self, X, t):
    while len ( t.shape ) < len ( X.shape ):
      t = t.unsqueeze(-1)

    t = t.to(X)

    print ( X.shape, t.shape )
    return self.model(torch.cat([X, t.expand_as(X)], dim=-1)).reshape_as(X)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CIFARMLP().to(device)

trainData = cifarTrainset ( 128 )

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

# Overfit a single batch,
# see if this works

x = (next(iter(trainData))[0]).to(device)
print(x)

noiseSchedule = CIFARProcess()
# noiseSchedule.visualizeForward(x[0])
# noiseSchedule.visualizeReverse(x[0])

for i in range(10000):
  noiseDict = noiseSchedule.noiseForward(x)

  noisedX = noiseDict["noiseX"]
  priorMean = noiseDict["XMeanOneDown"]
  destinationTimesteps = ( noiseDict["timesteps"] - 1 ) / noiseSchedule.steps

  reconLoss = nn.functional.mse_loss ( model(noisedX.float(), destinationTimesteps.float()), priorMean.float() )

  print ( reconLoss )

  optim.zero_grad()
  reconLoss.backward()
  optim.step()

  if i % 1000 == 0 and i > 1:
    # Sample an image, using all 1000 timesteps
    randX = torch.randn_like(x[0]).unsqueeze(0)

    for it in range(999, -1, -1):
      tStepTensor = torch.tensor([it]).to(randX) / noiseSchedule.steps
      randX = noiseSchedule.reverseMean(randX, model(randX.float(), tStepTensor.float()), it)

    plt.imshow(randX[0].detach().cpu().numpy().transpose(1,2,0))
    plt.savefig(f"rand_{i}.png")
