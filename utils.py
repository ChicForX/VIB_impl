import torch
import sklearn.linear_model
import sklearn.metrics as metrics
from mine import MINE
from config import config_dict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import warnings
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from torchvision.utils import save_image
import os

eval_dir = config_dict['eval_dir']
sample_dir = config_dict['sample_dir']


def show_images(loader):
    images, labels, colors = next(iter(loader))

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        # [height,width,channel]
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

# privacy prediction
def evaluate_private_representations(encoder, train_dataset, test_dataset, device):
    encoder = encoder.to(device).eval()
    # train linear regression
    X_train, S_train = train_dataset.data, train_dataset.hidden
    z_train, _ = encoder(torch.FloatTensor(X_train).to(device))
    z_train = z_train.detach().cpu().numpy()

    s_predictor = sklearn.linear_model.LogisticRegression()
    s_predictor.fit(z_train, S_train)

    # test
    X_test, S_test = test_dataset.data, test_dataset.hidden
    z_test, _ = encoder(torch.FloatTensor(X_test).to(device))
    z_test = z_test.detach().cpu().numpy()

    s_pred_prob = s_predictor.predict_proba(z_test)
    # accuracy
    accuracy_s = metrics.accuracy_score(S_test, s_pred_prob)
    print(f'Accuracy on S (Logistic Regression): {accuracy_s}')

    return accuracy_s


# save output of sampled z
def save_imgs(x, model, epoch):
    out, _ = model.encoder(x)
    # Splice input and output together, output and save
    # batch_size*1*28*（28+28）=batch_size*1*28*56
    x_concat = torch.cat([x.view(-1, 3, 28, 28), out.view(-1, 3, 28, 28)], dim=3)
    save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))


# Perform t-SNE on the latent variables
# & show reconstruction image of first and last epoch
def eval_tsne_image():
    fig, ax = plt.subplots(1, 3)
    eval_true_label = np.load(eval_dir + "/eval_true_label.npy")
    eval_pred_label = np.load(eval_dir + "/eval_pred_label.npy")
    plotdistribution(eval_true_label, eval_pred_label, ax)

    # Display reconst-1 and reconst-15 images
    image_1 = mpimg.imread(sample_dir + '/reconst-1.png')
    plt.subplot(1, 3, 2)
    ax[1].imshow(image_1)
    ax[1].set_axis_off()

    image_epoch = mpimg.imread(sample_dir + '/reconst-15.png')
    plt.subplot(1, 3, 3)
    ax[2].imshow(image_epoch)
    ax[2].set_axis_off()
    plt.show()

def plotdistribution(Label, Mat, ax):
    warnings.filterwarnings('ignore', category=FutureWarning)
    tsne = TSNE(n_components=2, random_state=0)
    Mat = tsne.fit_transform(Mat[:])

    x = Mat[:, 0]
    y = Mat[:, 1]
    map_color = {0: 'r', 1: 'g',2:'b',3:'y',4:'k',5:'m',6:'c',7:'pink',8:'grey',9:'blueviolet'}
    color = list(map(lambda x: map_color[x], Label))
    ax[0].scatter(np.array(x), np.array(y), s=5, c=color, marker='o')  # The scatter function only supports array type data
    ax[0].set_axis_on()

    # add labels
    legend_elements = []
    for label, color in map_color.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label))
    ax[0].legend(handles=legend_elements, title='Label', loc='upper right', handlelength=0.8, handleheight=0.8)

# train mine
def train_mine(model, mine, train_loader, epochs=20):
    print("MINE starts training.")

    optimizer = torch.optim.Adam(mine.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, u_batch, _ in train_loader:
            z_batch, _ = model.encoder(x_batch)
            optimizer.zero_grad()
            mi_estimate = mine(z_batch, u_batch)
            loss = -torch.mean(mi_estimate)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, MI Estimate: {-total_loss / len(train_loader)}")




