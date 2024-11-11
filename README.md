# DAT494FinalProject
My final project for 494, which is mostly about conditional generation on CIFAR-10.

I'm going to be using GANs, DDPM for Diffusion, and also characterizing their diversity
using a combination of inception score and an auto-encoder reconstruction analysis;
the better the reconstruction, the less diverse the underlying generating distribution
probably is. I expect the GAN to lose here, but we'll see.

As well, since this is a conditional generation task, I'll also report accuracy of both
methods i.e. what percentage of their generated samples match the correct class. I'm going
to be using a RESNET 50 based classifier for this, which will also provide classifier guidance
to the diffusion model.
