import torch
from torchvision import datasets, transforms
import numpy as np
from config import config_dict

batch_size = config_dict['batch_size']

class MNIST_dataset(torch.utils.data.Dataset):
    def __init__(self, data, class_labels, color_labels, transform=None, task=None):
        self.data = data
        self.class_labels = class_labels
        self.color_labels = color_labels
        self.transform = transform
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        class_label = self.class_labels[idx]
        color_label = self.color_labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.task == 'privacy':
            # Privacy-related processing
            pass

        return image, class_label, color_label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_data = datasets.MNIST(root='./data/MNIST/train', train=True, download=True)
test_data = datasets.MNIST(root='./data/MNIST/test', train=False, download=True)

# Function to randomly assign color channels
def assign_color_channels(data, color_labels):
    num_samples = data.shape[0]
    idx = np.random.permutation(num_samples)
    split = num_samples // 3
    # Red channel
    data[idx[:split], 0, :, :] = data[idx[:split], 0, :, :]
    # Green channel
    data[idx[split:2*split], 1, :, :] = data[idx[split:2*split], 1, :, :]
    # Blue channel
    data[idx[2*split:], 2, :, :] = data[idx[2*split:], 2, :, :]
    color_labels[idx[:split]] = 0
    color_labels[idx[split:2*split]] = 1
    color_labels[idx[2*split:]] = 2

# Add color channels
train_images = np.repeat(train_data.data.numpy()[:, np.newaxis, :, :], 3, axis=1)
test_images = np.repeat(test_data.data.numpy()[:, np.newaxis, :, :], 3, axis=1)

# Privacy labels (colors)
s_train = np.zeros(len(train_images), dtype=np.uint8)
s_test = np.zeros(len(test_images), dtype=np.uint8)

assign_color_channels(train_images, s_train)
assign_color_channels(test_images, s_test)

# Class labels
u_train = train_data.targets.numpy()
u_test = test_data.targets.numpy()

# Convert numpy to tensor
x_train_tensor = torch.tensor(train_images, dtype=torch.float32)
x_test_tensor = torch.tensor(test_images, dtype=torch.float32)
s_train_tensor = torch.tensor(s_train, dtype=torch.long)
s_test_tensor = torch.tensor(s_test, dtype=torch.long)
u_train_tensor = torch.tensor(u_train, dtype=torch.long)
u_test_tensor = torch.tensor(u_test, dtype=torch.long)

# Create custom dataset
train_dataset = MNIST_dataset(x_train_tensor, u_train_tensor, s_train_tensor, transform=transform, task='privacy')
test_dataset = MNIST_dataset(x_test_tensor, u_test_tensor, s_test_tensor, transform=transform, task='privacy')

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
