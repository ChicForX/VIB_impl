import torch
from torchvision import datasets, transforms
import numpy as np
from config import config_dict
from utils import show_mnist_images

batch_size = config_dict['batch_size']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST
train_data = datasets.MNIST(root='./data/MNIST/train', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data/MNIST/test', train=False, download=True, transform=transform)

train_images = train_data.data.numpy()
test_images = test_data.data.numpy()

# add color channels
# [batch_size, color, height, width]
x_train = np.zeros((train_images.shape[0], 3, train_images.shape[1], train_images.shape[2]), dtype=np.uint8)
x_test = np.zeros((test_images.shape[0], 3, test_images.shape[1], test_images.shape[2]), dtype=np.uint8)

# color labels: sensitivity
s_train = np.zeros(train_images.shape[0], dtype=np.uint8)
s_test = np.zeros(test_images.shape[0], dtype=np.uint8)

# class labels: utility
u_train = train_data.targets.numpy()
u_test = test_data.targets.numpy()


def assign_color_channels(x_data, original_data, color_labels):
    num_samples = x_data.shape[0]
    idx = np.random.permutation(num_samples)
    split = num_samples // 3

    # float type for color
    x_data = x_data.astype(np.float32)

    for i in range(split):
        x_data[idx[i], 0, :, :] = original_data[idx[i]]  # red
        color_labels[idx[i]] = 0
    for i in range(split, 2 * split):
        x_data[idx[i], 1, :, :] = original_data[idx[i]]  # green
        color_labels[idx[i]] = 1
    for i in range(2 * split, num_samples):
        x_data[idx[i], 2, :, :] = original_data[idx[i]]  # blue
        color_labels[idx[i]] = 2

    x_data /= 255.0
    return x_data, color_labels

x_train, s_train = assign_color_channels(x_train, train_images, s_train)
x_test, s_test = assign_color_channels(x_test, test_images, s_test)

# numpy to tensor
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
s_train_tensor = torch.tensor(s_train, dtype=torch.long)
s_test_tensor = torch.tensor(s_test, dtype=torch.long)
u_train_tensor = torch.tensor(u_train, dtype=torch.long)
u_test_tensor = torch.tensor(u_test, dtype=torch.long)

# dataset
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, u_train_tensor, s_train_tensor)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, u_test_tensor, s_test_tensor)

# data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# draw origin data
#show_mnist_images(test_loader)
