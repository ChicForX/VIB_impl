import torch
from nets import VAE
from config import config_dict
from datasets import train_loader, test_loader


# hyperparams
epochs = config_dict['epochs']
lr = config_dict['lr']
batch_size = config_dict['batch_size']
dim_img = config_dict['dim_img']
dim_input = dim_img * dim_img * 3
dim_z = config_dict['dim_z']
dim_s = config_dict['dim_s']

vae = VAE(dim_input, dim_z, sensitive_dim=dim_s)
optimizer = torch.optim.Adam(VAE.parameters(), lr=lr)

def train():
    for epoch in range(epochs):
        for x, u, s in train_loader:
            x = x.view(-1, dim_input)
            optimizer.zero_grad()

            z, (z_mu, z_logvar) = vae.encoder(x, s)
            x_hat = vae.decoder(z, s)

            loss = vae.loss_function(x_hat, x, z_mu, z_logvar)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


if __name__ == "__main__":
    train()
    # TODO test
