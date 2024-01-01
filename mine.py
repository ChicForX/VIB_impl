import torch
import torch.nn as nn
import math


class MINE(nn.Module):
    def __init__(self, moving_average_rate=0.01, moving_average_expT=1.0):
        super(MINE, self).__init__()
        # representation Z(batch_size, 3, 28, 28)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # S or U
        self.fc_s = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        # concat S/U and Z
        self.fc_combined = nn.Sequential(
            nn.Linear(32 * 7 * 7 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.moving_average_rate = moving_average_rate
        self.moving_average_expT = moving_average_expT

    def forward(self, z, s):
        z = self.conv_layers(z)
        s = s.float()
        s = self.fc_s(s.unsqueeze(1))
        combined = torch.cat((z, s), 1)
        return self.fc_combined(combined)

    def get_mi(self, z, s):
        t = self(z, s).mean()
        s_tilde = s[torch.randperm(s.size(0))]
        expt = torch.exp(self(z, s_tilde)).mean()
        mi = (t - torch.log(expt)).item() / math.log(2)
        return mi

    # remember to node model.train() outside the func
    def train_mine_inside_epoch(self, z, s, optimizer):
        optimizer.zero_grad()
        # T(z,s)
        t = self(z, s)
        # shuffle z
        t_shuffled = self(z, s[torch.randperm(s.size(0))])
        T = t.mean()
        expT = torch.exp(t_shuffled).mean()
        self.moving_average_expT = (1 - self.moving_average_rate) * self.moving_average_expT + self.moving_average_rate * expT.item()

        # Donsker-Varadhan
        mi_loss = -(T - expT / self.moving_average_expT)
        mi_loss.backward()
        optimizer.step()

        return mi_loss.item()

