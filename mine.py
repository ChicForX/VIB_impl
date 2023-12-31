import torch
import torch.nn as nn
import torch.optim as optim


class MINE(nn.Module):
    def __init__(self):
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

    def forward(self, z, s):
        z = self.conv_layers(z)
        s = s.float()
        s = self.fc_s(s.unsqueeze(1))
        combined = torch.cat((z, s), 1)
        return self.fc_combined(combined)

    # remember to node model.train() outside the func
    def train_mine_inside_epoch(self, z, s, optimizer):
        optimizer.zero_grad()
        # T(z,s)
        t = self(z, s)
        # shuffle z
        t_shuffled = self(z, s[torch.randperm(s.size(0))])
        # Donsker-Varadhan
        mi_loss = -(torch.mean(t) - torch.log(torch.mean(torch.exp(t_shuffled))))
        mi_loss.backward()
        optimizer.step()
        return mi_loss

