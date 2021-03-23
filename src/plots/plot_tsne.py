import pathlib

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_tsne(tx, ty, labels_val, class_target, class_out, title="", save=True, title_extra="", folder=""):
    """
    :param tx: x-coords from tsne
    :param ty: y-coords from tsne
    :param labels_val: labels for coloring
    :param title: title for plot and saving
    :param save: boolean for saving
    :param title_extra: give an additional title for saving
    :param folder: sub folder for saving
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # labels_val = torch.flip(labels_val, [0])
    labels = ["$D_T$: " + class_target, "$D_A$: " + str(class_out)]
    colors = ['#30828A', '#BD5C23']
    # colors = ['#1BA893', '#BD5C23']

    # for every class, we'll add a scatter plot separately
    for idx, label in enumerate(labels):
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels_val) if l == idx]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format
        color = colors[idx]

        # add a scatter plot with the corresponding color and label
        if label == "ref_data":
            ax.scatter(current_tx, current_ty, c=color, alpha=0.7, label=label)
        else:
            ax.scatter(current_tx, current_ty, c=color, label=label)


    # build a legend using the labels we set previously
    ax.legend(loc='best')
    plt.xticks([])
    plt.yticks([])
    if title == "":
        title = str(np.random.randint(0, 1000))
    if save:
        pathlib.Path('../export/tsne/' + folder).mkdir(parents=True,exist_ok=True)
        plt.savefig("../export/tsne/" + folder + title + title_extra + ".svg")
    plt.show()


if __name__ == "__main__":
    tx = np.load("../tsne_dummidata/tx.npy")
    ty = np.load("../tsne_dummidata/ty.npy")
    labels_val = np.load("../tsne_dummidata/labels.npy")
    plot_tsne(tx, ty, labels_val, save=False)
