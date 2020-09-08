import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F


class LFAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config['rgb']:
            self.C = 3
        else:
            self.C = 1

        self.D = self.config['D']
        self.H = self.config['H']
        self.W = self.config['W']

        self.beta = 0.001
        self.batch_size = self.config['batch_size']
        self.features = config['features']

        # change here , make not hardcoded!!!!!!!
        self.fc_mu = nn.Linear(3 * 3 * 3 * 196, 3 * 3 * 3 * 196)
        self.fc_var = nn.Linear(3 * 3 * 3 * 196, 3 * 3 * 3 * 196)
        self.encoder = Encoder(self.features)

    def encode(self, input):
        res = self.encoder(input)
        res = torch.flatten(res, start_dim=1)
        mu = self.fc_mu(res)
        var = self.fc_var(res)
        return mu, var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input1, input2):
        mu1, var1 = self.encode(input1)
        mu2, var2 = self.encode(input2)
        z1 = self.reparameterize(mu1, var1)
        z2 = self.reparameterize(mu2, var2)


class Encoder(nn.Module):
    def __init__(self, features):
        super(Encoder, self).__init__()
        self.block1 = Block(features[0], features[1], first_block=True)
        self.block2 = Block(features[1], features[2])
        self.block3 = Block(features[2], features[3])
        self.block4 = Block(features[3], features[4])
        self.block5 = Block(features[4], features[5])
        self.block6 = Block(features[5], features[6])

        self.encoder = nn.Sequential(self.block1, self.block2, self.block3,
                                     self.block4, self.block5, self.block6)

    def forward(self, input):
        # import pdb;pdb.set_trace()
        out = self.encoder(input)
        return out


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, first_block=False):
        super(Block, self).__init__()
        self.first_block = first_block
        if self.first_block:
            self.layer1 = Layer(3, in_channel)
        else:
            self.layer1 = Layer(in_channel, in_channel)

        self.layer2 = Layer(in_channel, in_channel)
        self.layer3 = Layer(in_channel, out_channel, stride=2)
        self.block = nn.Sequential(self.layer1, self.layer2, self.layer3)

    def forward(self, input):
        # import pdb;pdb.set_trace()
        out = self.block(input)
        return out


class Layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(Layer, self).__init__()
        self.batch_norm = nn.BatchNorm3d(in_channel)
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size, stride)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        import pdb;
        pdb.set_trace()
        norm = self.batch_norm(input)
        conv = self.conv(norm)
        out = self.relu(conv)
        return out + input
