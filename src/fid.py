"""
Computes the FID score
for a model generating CIFAR
models.

On initial import, this module
computes CIFAR summary stats once,
to save time later. This does
increase load time, but can
fix this later by saving to disk
"""
import os

import matplotlib.pyplot as plt
import scipy.linalg
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

from torchmetrics.image.fid import FrechetInceptionDistance

import torchvision.transforms as transforms

# transform = transforms.Compose([
#     transforms.Resize((299, 299)),  # Resize to 299x299
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to [-1, 1] range
# ])
transform = Inception_V3_Weights.DEFAULT.transforms()

from cifar import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inception = inception_v3(pretrained=True).to(device)
inception.eval()

return_nodes = { "avgpool" : "features" }
extractor = create_feature_extractor(inception, return_nodes=return_nodes)

def getEmbeds( rawImgBatch ):
  """
  Simple function to get
  Inception embeddings for CIFAR-10
  sized images.
  """

  with torch.no_grad():
    return extractor( transform(rawImgBatch.to(device)) )["features"].squeeze()

class LambdaModule(nn.Module):
  def __init__(self, f):
    super(LambdaModule, self).__init__()
    self.f = f
 
  def forward(self, x):
    return self.f(x)

featModel = LambdaModule(getEmbeds)
fid = FrechetInceptionDistance(feature=featModel, normalize=True, reset_real_features = False ).to(device)

# Add all features from cifar10 right now
with torch.no_grad():
  cifarTrainset = cifarTrainset(512)
  for X, y in cifarTrainset:
    fid.update ( X.to(device), real=True )

# Good to go

# cifarTrainMean = None
# cifarTrainCovar = None
# 
# # File names where we store
# # CIFAR-10 train mean FID,
# # and covar across all classes
# allClassesCIFARTrainFIDMeanFName = "cifarTrainFIDMean_ALL.pt"
# allClassesCIFARTrainFIDCovarFName = "cifarTrainFIDCovar_ALL.pt"
# 
# if allClassesCIFARTrainFIDMeanFName in os.listdir("."):
#   cifarTrainMean = torch.load( allClassesCIFARTrainFIDMeanFName )
#   cifarTrainCovar = torch.load( allClassesCIFARTrainFIDCovarFName )
# 
# if cifarTrainMean is None:
#   BS = 500
#   trainSet = cifarTrainset ( BS )
# 
#   # Maintain a list of batch
#   # means here. As long as
#   # all batches have the same
#   # size, average of average
#   # is the same as a single average.
#   averages = []
#   covariances = []
# 
#   for X, _ in trainSet:
#     if X.shape[0] != BS:
#       break
#     
#     embeds = getEmbeds ( X )
#     averages.append ( embeds.mean(dim=0 ) )
# 
#     
#   # Now, take means of averages and covariance. That's the final one
#   cifarTrainMean = torch.stack( averages ).mean(dim=0)
# 
#   # Now, iterate back over, and compute
#   # the covar, subtracting embeds
#   # by the global mean
#   for X, _ in trainSet:
#     if X.shape[0] != BS:
#       break
# 
#     embeds = getEmbeds( X ) - cifarTrainMean
#     covariances.append ( torch.matmul( embeds.T, embeds ) )
# 
#   cifarTrainCovar = torch.stack( covariances ).mean(dim=0)
# 
#   # Store to disk
#   torch.save( cifarTrainMean, allClassesCIFARTrainFIDMeanFName )
#   torch.save( cifarTrainCovar, allClassesCIFARTrainFIDCovarFName )
# 
# print ( f"CIFAR-10 Train Mean: {cifarTrainMean.shape}" )
# print ( f"CIFAR-10 Train Covar: {cifarTrainCovar.shape}" )
# 
# def computeFID_Uncond(genFunction):
#   """
#   Given a generation function
#   that returns image batches
#   of some shape, compute FID
#   over 1000 generated images.
# 
#   This is for the unconditional case.
#   """
#   genMeans = []
#   genCovars = []
#   numSamples = 0
# 
#   with torch.no_grad():
#     while numSamples < 1000:
#       nextGen = genFunction()
#       print("RAN1")
# 
#       embeds = getEmbeds( nextGen )
#       genMeans.append ( embeds.mean(dim=0) )
#       
#       numSamples += nextGen.shape[0]
# 
#     totalMean = torch.stack( genMeans ).mean(dim=0)
# 
#     # Now, iterate back over and compute
#     # covariance
#     numSamples = 0
#     while numSamples < 1000:
#       print("RAN2")
#       nextGen = genFunction()
#       embeds = getEmbeds( nextGen ) - totalMean
#       genCovars.append ( torch.matmul( embeds.T, embeds ) )
#       numSamples += nextGen.shape[0]
# 
#     totalCovar = torch.stack( genCovars ).mean(dim=0)
# 
#   print ( totalMean )
#   print ( totalCovar )
# 
#   print ( f"Total Mean GAN: {totalMean}")
#   print ( f"Total Covar GAN: {totalCovar}")
# 
#   print(f"Total Mean CIFAR: {cifarTrainMean}")
#   print(f"Total Covar CIFAR: {cifarTrainCovar}")
# 
#   print(f"COVAR's matmul: {torch.matmul( cifarTrainCovar, totalCovar )}")
# 
#   # sqrtm isn't supported in torch, so use scipy
#   # and then cast back
#   sqrtMatProd = torch.tensor(scipy.linalg.sqrtm( torch.matmul( cifarTrainCovar, totalCovar ).numpy(force=True) ).real).to(device)
# 
#   print ( f" Mean diference: {torch.norm( cifarTrainMean - totalMean )**2}" )
#   print ( f"Trace: {torch.trace( cifarTrainCovar + totalCovar - 2*sqrtMatProd )}" )
#   fid = torch.norm( cifarTrainMean - totalMean )**2 + torch.trace( cifarTrainCovar + totalCovar - 2*sqrtMatProd )
# 
#   print(f"Computed FID: {fid}")
#   return fid
