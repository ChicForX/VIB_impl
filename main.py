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

# mutual information estimation
mine1 = MINE().to(device)
mine2 = MINE().to(device)
optimizer1 = torch.optim.Adam(mine1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(mine2.parameters(), lr=0.001)
mi_z_s_values = []
mi_z_u_values = []

def train():
    for epoch in range(epochs):
        u_np = []
        u_hat_np = []
        total_mi_z_s = 0
        total_mi_z_u = 0
        for itrt, (x, u, s) in enumerate(train_loader):
            vae.train()
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
                                              cndtn_entropy_loss.item(), ce_loss.item(), kl_div.item()))
                # save for T-SNE
                if not os.path.exists(eval_dir):
                    os.mkdir(eval_dir)
                if os.path.exists(eval_dir + "/eval_true_label.npy"):
                    os.remove(eval_dir + "/eval_true_label.npy")
                if os.path.exists(eval_dir + "/eval_pred_label.npy"):
                    os.remove(eval_dir + "/eval_pred_label.npy")
                np.save(eval_dir + "/eval_true_label.npy", u_np)
                np.save(eval_dir + "/eval_pred_label.npy", u_hat_np)

            vae.eval()
            if epoch <= 4:
                mine1.train()
                mine2.train()
                mine1.train_mine_inside_epoch(z.detach(), s.detach(), optimizer1)
                mine2.train_mine_inside_epoch(z.detach(), u.detach(), optimizer2)
            total_mi_z_s += mine1.get_mi(z.detach(), s.detach())
            total_mi_z_u += mine2.get_mi(z.detach(), u.detach())


        if epoch > 4:
            mi_z_s_values.append(total_mi_z_s / itrt)
            mi_z_u_values.append(total_mi_z_u / itrt)
            print(f"Epoch {epoch + 1}, I(Z, S): {total_mi_z_s / itrt}, I(Z, U): {total_mi_z_u / itrt}")
        vae.eval()
        with torch.no_grad():
            # save imgs of each epoch
            utils.save_imgs(x, vae, epoch)


if __name__ == "__main__":
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    train()
    utils.eval_tsne_image(epochs, train_loader)
    utils.plot_mi(mi_z_s_values, mi_z_u_values)
