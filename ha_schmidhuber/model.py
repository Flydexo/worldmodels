import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
        
class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("hey", x.shape)
        return x

class Dense(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mu = nn.Linear(1024, 32)
        self.std = nn.Linear(1024, 32)
        self.cfg = cfg

    def forward(self, x):
        distribution = Normal(torch.zeros(x.shape[0], 32), torch.ones(x.shape[0], 32))
        x = x.flatten(start_dim=1)
        return self.mu(x) + self.std(x)*distribution.sample().to(self.cfg.device)


class Fit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(-1).unsqueeze(-1)
        
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(),
            Dense(cfg),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 1024),
            Fit(),
            nn.ConvTranspose2d(1024, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))
        