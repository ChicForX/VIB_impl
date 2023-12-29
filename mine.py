import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, dim_x, dim_z):
        super(MINE, self).__init__()
        self.fc1 = nn.Linear(dim_x + dim_z, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, y):
        # Concatenate x and y
        joint = torch.cat((x, y), dim=1)
        h = torch.relu(self.fc1(joint))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)

