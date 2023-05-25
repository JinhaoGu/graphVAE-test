import torch
from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from torch import nn
import torch.nn.functional as F

class GraphVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphVAE, self).__init__()
        
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv_mu = GCNConv(out_channels, out_channels)
        self.conv_logvar = GCNConv(out_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            return mu + torch.randn_like(std) * std
        else:
            return mu
    
    def decode(self, z, edge_index):
        return self.conv1(z, edge_index)
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, edge_index), mu, logvar
