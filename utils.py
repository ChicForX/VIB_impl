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


def show_mnist_images(loader):
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


def extract_samples_from_loader(data_loader, num_samples=1000):
    images_list = []
    colors_list = []
    accumulated_samples = 0

    for images, _, colors in data_loader:
        batch_size = images.size(0)
        images_list.append(images)
        colors_list.append(colors)
        accumulated_samples += batch_size
        if accumulated_samples >= num_samples:
            break

    images = torch.cat(images_list, dim=0)
    colors = torch.cat(colors_list, dim=0)
    return images, colors

# Perform t-SNE on the latent variables
# & show reconstruction image of first and last epoch
def eval_tsne_image(epoch, train_loader, num_samples=1000):
    fig, axs = plt.subplots(1, 3)

    # tsne of mnist before training
    images, colors = extract_samples_from_loader(train_loader, num_samples)
    num_samples = min(num_samples, images.shape[0])
    data = images[:num_samples].view(num_samples, -1).numpy()
    labels = colors[:num_samples].numpy()
    plotdistribution(labels, data, [axs[0]], map_color={0: 'red', 1: 'green', 2: 'blue'})

    # tsn of mnist after training
    eval_true_label = np.load(eval_dir + "/eval_true_label.npy")
    eval_pred_label = np.load(eval_dir + "/eval_pred_label.npy")
    plotdistribution(eval_true_label, eval_pred_label, [axs[1]],
                     map_color={0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'k', 5: 'm', 6: 'c', 7: 'pink', 8: 'grey',
                                9: 'blueviolet'})

    # Display reconst-15 images
    image_epoch = mpimg.imread(sample_dir + '/reconst-' + str(epoch) + '.png')
    plt.subplot(1, 3, 3)
    axs[2].imshow(image_epoch)
    axs[2].set_axis_off()
    plt.show()


def plotdistribution(Label, Mat, ax, map_color):
    warnings.filterwarnings('ignore', category=FutureWarning)
    tsne = TSNE(n_components=2, random_state=0)
    Mat = tsne.fit_transform(Mat[:])

    x = Mat[:, 0]
    y = Mat[:, 1]

    color = list(map(lambda x: map_color[x], Label))
    ax[0].scatter(np.array(x), np.array(y), s=5, c=color,
                  marker='o')  # The scatter function only supports array type data
    ax[0].set_axis_on()

    # add labels
    legend_elements = []
    for label, color in map_color.items():
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label))
    ax[0].legend(handles=legend_elements, title='Label', loc='upper right', handlelength=0.8, handleheight=0.8)

def plot_mi(mi_z_s_values, mi_z_u_values):
    total_iterations = len(mi_z_s_values)
    x_axis = list(range(total_iterations))
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, mi_z_s_values, label='I(Z, S)')
    plt.plot(x_axis, mi_z_u_values, label='I(Z, U)')
    plt.xlabel('Epochs')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information Across Epochs')
    plt.legend()
    plt.show()