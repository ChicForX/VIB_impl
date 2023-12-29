import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=5, padding=2),
            nn.BatchNorm2d(5),
            nn.ReLU6(inplace=True),
            nn.Conv2d(5, 50, kernel_size=5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU6(inplace=True),
            nn.Conv2d(50, 3, kernel_size=5, padding=2),
        )

        # logvar
        self.z_logvar = torch.nn.Parameter(torch.Tensor([-1.0]))

    def reparameterize(self, z_mu, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, x):
        z_mu = self.layers(x)
        z = self.reparameterize(z_mu, self.z_logvar)
        return z, z_mu


class Decoder(nn.Module):
    def __init__(self, dim_s):
        super(Decoder, self).__init__()
        # conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # prediction
        self.fc1 = nn.Linear(64 * 7 * 7 + dim_s, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, z, s):
        z = F.relu(self.conv1(z))
        z = self.pool(z)
        z = F.relu(self.conv2(z))
        z = self.pool(z)

        z = z.view(z.size(0), -1)
        s_expanded = s.unsqueeze(1)
        z = torch.cat((z, s_expanded), dim=1)

        z = F.relu(self.fc1(z))
        output = self.fc2(z)
        return output

class VAE(nn.Module):
    def __init__(self, dim_s):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(dim_s)

    def get_kl(self, z_mu):
        log_var = 0.5 * self.encoder.z_logvar
        kl_div = -0.5 * torch.sum(1.0 + log_var - torch.pow(z_mu, 2) - torch.exp(log_var))
        return kl_div / math.log(2)  # in bits

    def get_ce(self, output, u):
        u = u.view(-1).long()
        CE = nn.functional.cross_entropy(output, u, reduction='sum')
        return CE

