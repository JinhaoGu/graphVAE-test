import torch
import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar,beta=0.01):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

