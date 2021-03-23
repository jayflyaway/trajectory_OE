import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_transforms = transforms.Compose([transforms.Resize(240, interpolation=2),
                                       transforms.CenterCrop(240),
                                       transforms.ToTensor()])

train_transforms_ruff = transforms.Compose(
    [transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
     transforms.Normalize((0.491373, 0.482353, 0.446667), (0.247059, 0.243529, 0.261569))])

test_transforms_ruff = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.491373, 0.482353, 0.446667), (0.247059, 0.243529, 0.261569))])


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=train_transforms):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        # ndarray should be in datasets.data.array
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = Image.fromarray(self.datasets[class_label][index_wrt_class])
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


class JJNet(models.AlexNet):
    def extract_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        layers = list(self.classifier.children())[:-2]
        for m in layers:
            x = m(x)
        return x


class CIFAR10_LeNet(torch.nn.Module):
    def __init__(self, rep_dim=256, bias_terms=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder network
        self.conv1 = nn.Conv2d(3, 32, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=bias_terms)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=bias_terms)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=bias_terms, padding=2)
        nn.init.xavier_normal_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=bias_terms)
        self.fc1 = nn.Linear(128 * 4 * 4, 512, bias=bias_terms)
        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn1d1 = nn.BatchNorm1d(512, eps=1e-04, affine=bias_terms)
        self.fc2 = nn.Linear(512, self.rep_dim, bias=bias_terms)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        x = F.leaky_relu(self.bn1d1(x))
        x = self.fc2(x)
        return x

# classifier = nn.Sequential(*list(mdl.classifier.children())[:-2])
# cross-entropy loss
def desc_loss(outputs, labels):
    # labels = torch.zeros(outputs.shape[0], dtype=torch.int64, device=outputs.device)
    d_loss = F.cross_entropy(outputs, labels)
    return d_loss


# what they wrote in the text:
def compactness_loss(outputs):
    # mean squared intra-batch distance
    n, m = outputs.shape
    c_loss = torch.zeros(n, dtype=outputs.dtype, device=outputs.device)

    for i in range(0, n):
        x = outputs[i, :]
        # x = outputs[i, :].clone().detach()
        x_others = torch.cat([outputs[0:i], outputs[i + 1:]])
        # m_i = mean_vector of the others
        mean_vec = torch.mean(x_others, dim=0)
        # z_i =  x_i - m_i
        z = x - mean_vec
        # c_loss[i] += np.math.pow((outputs.data[i][j] - mean_vec[j]) / float(n), 2)
        c_loss[i] = torch.sum(z * z)  # zt * z
        # c_loss[i] += torch.pow(z[j] / n, 2)
    c_loss = c_loss / m
    # c_loss = torch.mean(c_loss, dim=0)
    return c_loss


def get_features(model, loader, relabel=False, relabel_as_outliers=False, label_plus_one=False):
    """
    :param model: torch.model
    :param loader: data loader
    :param relabel: change labels which are not 0 to 1
    :param relabel_as_outliers: change all labels to 1
    :return:
    """
    # extract calculate features of this sample
    # return features
    outputs_features = []
    all_lables = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            inputs, labels = batch
            outputs_features.append(model.extract_features(batch[0]))
            all_lables.append(batch[1])
    outputs_features = torch.cat(outputs_features)
    all_lables = torch.cat(all_lables)
    if label_plus_one:
        all_lables = all_lables + torch.ones(len(all_lables), dtype=int)
    if relabel:
        all_lables[all_lables != 0] = 1  # set labels of other class to 1
    if relabel_as_outliers:
        all_lables[:] = 1
    return outputs_features, all_lables


def print_images(images):
    fig = plt.figure(figsize=(7., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(7, 8),
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    # Iterating over the grid returns the Axes.
    for ax, im in zip(grid, images):
        # normalize
        img = (im - im.min()) / (im.max() - im.min())
        ax.imshow(img.T, cmap='coolwarm')
    plt.show()
