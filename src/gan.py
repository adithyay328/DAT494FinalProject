import random
import os

import matplotlib.pyplot as plt

import wandb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchmetrics.image.fid import FrechetInceptionDistance

from cifar import *
from fid import fid, featModel

nz = 100 # Size of latent space, possibly with class concated
ngf = 64 # Size of feature maps in generator
ndf = 128 # Size of feature maps in discriminator
nc = 3 # Number of channels in our input

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
   
      # input is Z, going into a convolution
      nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
      # nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. ``(ngf*8) x 4 x 4``
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. ``(ngf*4) x 8 x 8``
      nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. ``(ngf*2) x 16 x 16``
      nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. ``(ngf) x 32 x 32``
      nn.Conv2d(ngf, 3, 1),
      nn.Tanh()
      # state size. ``(nc) x 64 x 64``
    )

  def forward(self, input):
    # If input is less than 4 long,
    # unsqueeze last dims
    while len(input.shape) < 4:
      input = input.unsqueeze(-1)

    return self.main(input)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
        nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. ``(ndf*2) x 16 x 16``
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. ``(ndf*4) x 8 x 8``
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. ``(ndf*8) x 4 x 4``
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
      )

  def forward(self, input):
    return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Test if it works
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rand = torch.randn(1, nz).to(device)
gen = Generator().to(device)
disc = Discriminator().to(device)

weights_init(gen)
weights_init(disc)

print ( gen(rand).shape )
print ( disc(gen(rand)).shape )

# if "uncond_generator.pth" in os.listdir():
#   gen.load_state_dict(torch.load("uncond_generator.pth"))
# if "uncond_discriminator.pth" in os.listdir():
#   disc.load_state_dict(torch.load("uncond_discriminator.pth"))

if __name__ == "__main__":
  hypers = {
      "lr": 0.0002,
      "architecture": "DCGAN, Unconditional",
      "dataset": "CIFAR-10",
      "epochs": 100,
      "BS" : 256
  }

  # Now, our train loop. Train discriminator first,
  # the generator.
  genOptim = optim.Adam(gen.parameters(), lr=hypers["lr"], betas=(0.5, 0.999))
  discOptim = optim.Adam(disc.parameters(), lr=hypers["lr"] / 4, betas=(0.5, 0.999))

  wandb.init(
    project="494Final",
    config=hypers
  )

  BS = hypers["BS"]
  trainLoader = cifarTrainset ( BS )
  
  realLabel = 1
  fakeLabel = 0
  
  import os

  for epoch in range(hypers["epochs"]):
    currEpochDiscriminatorAcc = []
    currEpochGeneratorAcc = []

    for x, y in trainLoader:
      # Zero out both grads
      genOptim.zero_grad()
      discOptim.zero_grad()
  
      # Train discriminator with all-reals
      realDiscPreds = disc(x.to(device) * 2 - 1)
      realGT = torch.ones_like(realDiscPreds).to(device) * realLabel
      loss = torch.nn.functional.binary_cross_entropy(realDiscPreds, realGT)
      correctTensor_DiscReal = ( realDiscPreds.round() == realGT.float() ).float()
      loss.backward()

      oldLoss = loss
  
      rand = torch.randn(BS, nz, 1, 1, device=device)
  
      fakeDiscPreds = disc(gen(rand).detach())
      fakeGT = torch.ones_like(fakeDiscPreds).to(device) * fakeLabel
      loss = torch.nn.functional.binary_cross_entropy(fakeDiscPreds, fakeGT)
      correctTensor_DiscFake = ( fakeDiscPreds.round() == fakeGT.float() ).float()
      loss.backward()

      newLoss = loss

      print(f"Total discriminator loss: { 1/2 * (oldLoss + newLoss) }")
  
      discOptim.step()

      # To compute average accuracy this iteration,
      # concatenate the two tensors, then take the mean
      currBatchDiscriminatorAcc = torch.mean(torch.cat([correctTensor_DiscReal, correctTensor_DiscFake]))

      currEpochDiscriminatorAcc.append( currBatchDiscriminatorAcc.item() )
  
      # Now, generator
      genOptim.zero_grad()
      discOptim.zero_grad()
  
      newRand = torch.randn(BS, nz, 1, 1, device=device)
      discPred = disc(gen(newRand))

      generatorFoolRate = torch.mean( (discPred.round() == realLabel).float() ).item()
      currEpochGeneratorAcc.append(generatorFoolRate)
  
      loss = -1 * torch.log(discPred).mean()
      loss.backward()

      print(f"Generator loss: {loss.item()}")
  
      genOptim.step()

    # Make 4 images, and log to wandb
    fourRands = torch.randn(4, nz, 1, 1, device=device)
    generated = gen(fourRands)
    # Pemute to shape Batch, ..., channels
    generated = generated.cpu().permute(0, 2, 3, 1).numpy(force=True)

    asListOfImages = [ wandb.Image( img ) for img in generated ]

    # Also, generate 2000 images, and compute FID
    fid.reset()
    with torch.no_grad():
      for i in range(8):
        rands = torch.randn(256, nz, device=device)
        gens = gen(rands)
        fid.update(gens, real = False)

    fidVal = fid.compute()

    wandb.log({
      "discriminator_accuracy": torch.mean(torch.tensor(currEpochDiscriminatorAcc)).item(),
      "generator_accuracy": torch.mean(torch.tensor(currEpochGeneratorAcc)).item(),
      "generated_images" : asListOfImages,
      "fid" : fidVal
    })

  # Save both models, then log to wandb using an
  # artifact
  artifact = wandb.Artifact('models', type='model')

  torch.save(gen.state_dict(), "uncond_generator.pth")
  torch.save(disc.state_dict(), "uncond_discriminator.pth")

  artifact.add_file( local_path="uncond_generator.pth", name="Generator")
  artifact.add_file( local_path="uncond_discriminator.pth", name="Discriminator")
  artifact.save()

# # Import FID, and compute
# def genFunction():
#   """
#   Generate 256 samples at a time
#   """
#   randZ = torch.randn(256, nz, 1, 1, device=device)
#   generated = gen(randZ)
# 
#   return generated
# 
# # 
# # fid.computeFID_Uncond ( genFunction )
# 
# # Make 2048 samples
# with torch.no_grad():
#   for i in range(40):
#     samples = genFunction() / 2 + 0.5
#     fid.update ( samples, real = False)
# 
# print ( fid.compute() )
