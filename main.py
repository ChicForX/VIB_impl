import torch
import utils
from nets import VAE
from config import config_dict
from datasets import train_loader, test_loader
import os
from mine import MINE
import numpy as np


# hyperparams
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = config_dict['epochs']
lr = config_dict['lr']
batch_size = config_dict['batch_size']
dim_img = config_dict['dim_img']
dim_input = dim_img * dim_img * 3
dim_z = config_dict['dim_z']
dim_s = config_dict['dim_s']
dim_u = config_dict['dim_u']
alpha = config_dict['alpha']
eval_dir = config_dict['eval_dir']
sample_dir = config_dict['sample_dir']
mi_epochs = config_dict['mi_epochs']

vae = VAE(dim_s).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

def train():
    for epoch in range(epochs):
        u_np = []
        u_hat_np = []
        for itrt, (x, u, s) in enumerate(train_loader):
            x, u, s = x.to(device), u.to(device), s.to(device)
            optimizer.zero_grad()

            z, z_mu = vae.encoder(x)
            u_hat = vae.decoder(z, s)

            kl_div = 1e-2 * vae.get_kl(z_mu)
            ce_loss = alpha * vae.get_ce(u_hat, u)
            cndtn_entropy_loss = 60 * alpha * vae.get_conditional_entropy(u_hat, s)
            loss = kl_div + ce_loss - cndtn_entropy_loss
            loss.backward()
            optimizer.step()

            # Record the output of latent space
            u_cpu = u.cpu().detach().numpy()
            u_hat_cpu = u_hat.cpu().detach().numpy()
            u_np.extend(u_cpu)
            u_hat_np.extend(u_hat_cpu)

            if (itrt + 1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Conditional_entropy Loss: {:.4f}, Cross_entropy Loss: {:.4f}, "
                      "KL Div: {:.4f}".format(epoch + 1, epochs, itrt + 1, len(train_loader),
                                              cndtn_entropy_loss.item(), ce_loss.item(),kl_div.item()))
                # save for T-SNE
                if not os.path.exists(eval_dir):
                    os.mkdir(eval_dir)
                if os.path.exists(eval_dir + "/eval_true_label.npy"):
                    os.remove(eval_dir + "/eval_true_label.npy")
                if os.path.exists(eval_dir + "/eval_pred_label.npy"):
                    os.remove(eval_dir + "/eval_pred_label.npy")
                np.save(eval_dir + "/eval_true_label.npy", u_np)
                np.save(eval_dir + "/eval_pred_label.npy", u_hat_np)

        with torch.no_grad():
            # save imgs of each epoch
            utils.save_imgs(x, vae, epoch)

    utils.eval_tsne_image(epoch)


def tst(model, test_loader):
    vae.eval()

    # train MINE
    mine = MINE(dim_z, dim_u).to(device)
    utils.train_mine(model, mine, test_loader, mi_epochs)
    total_mi_s_z = 0
    total_mi_u_z = 0

    with torch.no_grad():
        for x, u, s in test_loader:
            z_test = model.encoder(x)
            # calculate mutual information
            # s & z
            total_mi_s_z += mine(z_test, s).mean().item()
            # u & z
            total_mi_u_z += mine(z_test, u).mean().item()

        avg_mi_s_z = total_mi_s_z / len(test_loader)
        avg_mi_u_z = total_mi_u_z / len(test_loader)
        print(f"Mutual Information of s & z: {avg_mi_s_z}")
        print(f"Mutual Information of u & z: {avg_mi_u_z}")


if __name__ == "__main__":
    utils.tsne_image_before_training(train_loader)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    train()
    # tst(vae, test_loader)
