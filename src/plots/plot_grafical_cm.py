import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_nice_circle(tn, fp, fn, tp, circle_r=1, title="title", auc=0.0, save=True, title_extra="", folder=""):
    """
    :param tn:
    :param fp:
    :param fn:
    :param tp:
    :param circle_r:
    :param title: title for plot and saving
    :param auc: display value underneath the circle
    :param save: boolean for saving
    :param title_extra give an additional title for saving
    :param folder: sub folder for saving
    """

    # true positives
    x_tp = []
    y_tp = []
    for i in range(0, tp):
        r = np.random.uniform(0, circle_r)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        x_tp.append(x)
        y_tp.append(y)

    # false positives
    x_fp = []
    y_fp = []
    for i in range(0, fp):
        r = np.random.uniform(0, -circle_r)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        x_fp.append(x)
        y_fp.append(y)

    # true negatives
    x_tn = []
    y_tn = []
    for i in range(0, tn):
        r = np.random.uniform(-circle_r - 0.1, - circle_r * 1.5)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        x_tn.append(x)
        y_tn.append(y)

    # false negatives
    x_fn = []
    y_fn = []
    for i in range(0, fn):
        r = np.random.uniform(circle_r + 0.1, circle_r * 1.5)
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        x_fn.append(x)
        y_fn.append(y)

    plt.figure(1, figsize=(5, 4))
    plt.xlim(-circle_r, circle_r)
    plt.ylim(-circle_r, circle_r)
    plt.clf()
    ax = plt.gca()
    colors_bg = ['#ddffa0', '#ddffa0', '#f0cfd6', '#f0cfd6']
    colors = ['#568c0f', '#d1084e', '#d1084e', '#568c0f']
    texts = ['TP=' + str(tp), 'FP=' + str(fp), 'TN=' + str(tn), 'FN=' + str(fn)]
    markers = ['o', 'X', 'o', 'X']

    circle_fill_outer = plt.Circle((0, 0), 1.52, color='#d1084e', alpha=.2)
    circle_white = plt.Circle((0, 0), 1.02, color='w', fill=True)
    circle_fill_inner = plt.Circle((0, 0), 1.02, color='#baff3a', alpha=.2)
    ax.add_artist(circle_fill_outer)
    ax.add_artist(circle_white)
    ax.add_artist(circle_fill_inner)
    tp_dots,  = ax.plot(x_tp, y_tp, 'o', ms=2, color='#568c0f', label='TP=' + str(tp))  # green in green
    fp_dots,  = ax.plot(x_fp, y_fp, 'x', ms=3, color='#d1084e', label='FP=' + str(fp))  # red in green
    tn_dots,  = ax.plot(x_tn, y_tn, 'o', ms=2, color='#d1084e', label='TN=' + str(tn))  # red in red
    fn_dots,  = ax.plot(x_fn, y_fn, 'x', ms=3, color='#568c0f', label='FN=' + str(fn))  # green in red
    ax.set_title(title)
    plt.xticks([])
    plt.yticks([])
    ax.legend(handles=[Line2D([0], [0], color=colors_bg[i], lw=8, label=texts[i], marker=markers[i], mfc=colors[i], ms=6) for i in range(4)
                       ], loc=(1.01, 0))
    results = 'tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn) + ' tp=' + str(tp)
    res_auc = "auc= " + str(round(auc, 4))
    # plt.figtext(.5, 0.01, results, fontsize=10, ha='center')
    plt.text(0, -2.2, res_auc,  ha='center')
    plt.subplots_adjust(right=0.75, bottom=0.18)
    plt.axvline(x=0, ymin=0.04, ymax=0.96, linewidth=1.5, color='#2e3a04', alpha=0.5)
    if save:
        pathlib.Path('../export/graphics_cm/' + folder).mkdir(parents=True, exist_ok=True)
        plt.savefig("../export/graphics_cm/" + folder + title + title_extra + ".svg")
    plt.show()


if __name__ == '__main__':
    tn = 283
    fp = 717
    fn = 116
    tp = 884
    circle_r = 1

    plot_nice_circle(tn, fp, fn, tp, circle_r, auc=0.6302349336057201, save=False)
