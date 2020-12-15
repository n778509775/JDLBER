#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    """ initialize weights of fully connected layer
    """
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight, gain=1)
        m.bias.data.zero_()
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Encoder(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_inputs),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_inputs, num_inputs),

            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_inputs, num_inputs))
        self.encoder.apply(init_weights)
    def forward(self, x):
        x = self.encoder(x)
        return x


# Decoder_a
class Decoder_a(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_a, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_inputs, num_inputs),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(num_inputs, num_inputs),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_inputs, num_inputs))
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Decoder_b
class Decoder_b(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_b, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_inputs, num_inputs),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(num_inputs, num_inputs),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_inputs, num_inputs))
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Decoder_c
class Decoder_c(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_c, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_inputs, num_inputs),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(num_inputs, num_inputs),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_inputs, num_inputs))
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

#classifier combine with autoencoder
class Discriminator(nn.Module):
    def __init__(self, num_inputs):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

