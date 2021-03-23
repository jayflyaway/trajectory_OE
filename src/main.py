import argparse
import csv
import logging
from datetime import datetime
import re

import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torchvision import datasets, models, transforms
from torch.utils import data

from tqdm import trange, tqdm
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import utils.OC_utils as OCutils
import pathlib

from utils.aux_func import Aux, MeterCollection
from plots.plot_grafical_cm import plot_nice_circle
from plots.plot_tsne import plot_tsne
from utils.glob import classDict, classList


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--num-epochs-train', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--action', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--model-test', type=str, default='model_alexCIFAR10_bs_100_ept_300_lr_0.01_ll_0.8')
    parser.add_argument('--loss-lambda', type=float, default=0.8)
    parser.add_argument('--class-target', type=str, default='plane')
    parser.add_argument('--classes-oe', default=['rocket', 'pickup_truck', 'mouse', 'lion', 'cattle', 'wolf', 'turtle',
                                                 'camel', 'whale', 'bus'], nargs='+', help='classes of OE dataset')
    parser.add_argument('--class-out', default=['cat'], nargs='+', help='classes of CIFAR10 not used during training.')
    return parser.parse_args()


def train(model, loader, oe_loader, optimizer, args):
    model.train()
    model_saved_name = "model_alexCIFAR10" + "_bs_" + str(args.batch_size) + "_ept_" + str(
        args.num_epochs_train) + "_lr_" + str(args.lr) + "_ll_" + str(args.loss_lambda)
    model = model.to(device)
    for epoch in trange(args.num_epochs_train):
        meters = MeterCollection("loss", "c_loss", "d_loss")
        for batch, oe_batch in zip(loader, oe_loader):
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            oe_batch[0] = oe_batch[0].to(device)
            oe_batch[1] = oe_batch[1].to(device)
            # OCutils.print_images(batch[0])
            outputs = model.extract_features(batch[0])
            outputs_rds = model(oe_batch[0])
            c_loss = OCutils.compactness_loss(outputs)
            c_loss = torch.mean(c_loss, dim=0)
            d_loss = OCutils.desc_loss(outputs_rds, oe_batch[1])
            optimizer.zero_grad()
            loss = d_loss + args.loss_lambda * c_loss
            # loss = d_loss
            loss.backward()
            optimizer.step()
            meters.update(loss=loss.item(), c_loss=c_loss.item(), d_loss=d_loss.item())
            tqdm.write(str(meters))
        logger.info("epoch={:4d} {}".format(epoch, meters))
    tqdm.write('Done')
    pathlib.Path('../export_models/' + folder + sub_folder_target + sub_folder_oe).mkdir(parents=True,
                                                                                         exist_ok=True)
    torch.save(model.state_dict(),
               '../export_models/' + folder + sub_folder_target + sub_folder_oe + model_saved_name + '.pt')


def OCSVM(features, features_test, model_name_test, labels_test=torch.tensor(0)):
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1, verbose=1)
    clf.fit(features)
    cm = 0
    # testing with more than one class
    if type(labels_test) == torch.Tensor:
        y_pred = clf.predict(features_test)
        # change label declaration
        labels_test[labels_test == 1] = -1  # outliers
        labels_test[labels_test == 0] = 1  # target class
        cm = metrics.confusion_matrix(labels_test, y_pred)
        print(cm)
        # target: low FP rate -> TP/(TP + FN) richtig klassifiziert / von allen richtigen
        # diff ist genau dann negativ wenn de beiden Werte sich unterscheiden
        diff = y_pred * np.array(labels_test)
        n_error = diff[diff == -1].size
        acc = 1 - n_error / features_test.shape[0]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        fpr, tpr, thresholds = metrics.roc_curve(np.array(labels_test), y_pred)
        # AUC is to deal with situations where you have a very skewed sample distribution
        auc = metrics.auc(fpr, tpr)
        if len(args.class_out) > 1:
            out_name = "rest"
        else:
            out_name = str(args.class_out[0])
        plot_nice_circle(tn, fp, fn, tp, title=model_name_test, auc=auc, save=True, title_extra=title_plot,
                         folder=folder + sub_folder_target + sub_folder_oe)
        pathlib.Path('../export/results_' + folder + 'results_target_' + args.class_target).mkdir(parents=True, exist_ok=True)
        with open('../export/results_' + folder + 'results_target_' + args.class_target + '/results_' + model_name_test + '.csv', 'a', encoding='utf-8') as f:
            f.write(
                f'{args.class_target}, {out_name}, {args.classes_oe}, {tn}, {fp}, {fn}, {tp}, {tpr[1]}, {fpr[1]}, {auc}\n')
    else:
        y_pred = clf.predict(features_test)
        n_error = y_pred[y_pred == -1].size
        acc = 1 - n_error / features_test.shape[0]
        fpr, tpr, thresholds = metrics.roc_curve(np.array(labels_test), y_pred)
        # AUC is to deal with situations where you have a very skewed sample distribution
        auc = metrics.auc(fpr, tpr)
    print(auc)


def logger_start(logger, args):
    pathlib.Path('../export_models/' + folder + sub_folder_target + sub_folder_oe).mkdir(parents=True, exist_ok=True)
    hdlr = logging.FileHandler('../export_models/' + folder + sub_folder_target + sub_folder_oe + 'logging_' + str(
        datetime.now(tz=None)).replace(' ', '') + '.log')
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("lambda is: " + str(args.loss_lambda))
    logger.info("---------------------")
    logger.info(args)
    logger.info("---------------------")


if __name__ == '__main__':
    args = parse_args()
    class_oe = args.classes_oe
    class_oe = list(filter(None, class_oe))
    class_out = args.class_out
    class_out = list(filter(None, class_out))
    # class_oe = [int(item) for item in args.classes_oe.split(',')]
    num_class_oe = len(class_oe)
    model = OCutils.JJNet()
    model.classifier[6] = nn.Linear(4096, num_class_oe)
    tnes_vis = True
    # re.sub('([\[\]\'\, ])', '', str(args.class_oe)) == carbird
    oe_name = re.sub('([\[\]\'\, ])', '', str(args.classes_oe))
    if len(args.class_out) > 1:
        title_plot = "_" + args.class_target + "_rest"
    else:
        title_plot = "_" + args.class_target + "_" + str(args.class_out[0])

    folder = "CIFAR10_oe_CIFAR100/"
    sub_folder_target = args.class_target + "/"
    sub_folder_oe = oe_name + "/"
    # Adam seems not to work: loss does not go deeper then ~ 2, 08
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=0, amsgrad=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    cifar_train = datasets.CIFAR10(root='../data', train=True, download=True, transform=OCutils.train_transforms)
    cifar_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=OCutils.train_transforms)
    cifar_train_oe = datasets.CIFAR100(root='../data', train=True, download=True, transform=OCutils.train_transforms)

    cifar100_class_dict = cifar_train_oe.class_to_idx

    # Separating trainset/testset data/label
    x = cifar_train.data
    y = cifar_train.targets
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(x, y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = np.array(y)[train_idx], np.array(y)[val_idx]
    # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test = cifar_test.data
    y_test = cifar_test.targets
    x_train_oe = cifar_train_oe.data
    y_train_oe = cifar_train_oe.targets

    # 1 class as target class
    cifar_train_target = OCutils.DatasetMaker(
        [OCutils.get_class_i(x_train, y_train, classDict[key]) for key in [args.class_target]],
        OCutils.train_transforms
    )
    cifar_train_oe = OCutils.DatasetMaker(
        [OCutils.get_class_i(x_train_oe, y_train_oe, cifar100_class_dict[key]) for key in class_oe],
        OCutils.train_transforms
    )
    cifar_val_target = OCutils.DatasetMaker(
        [OCutils.get_class_i(x_val, y_val, classDict[key]) for key in [args.class_target]],
        OCutils.train_transforms
    )
    # 1 class for abnormalities during testing
    cifar_val_out = OCutils.DatasetMaker(
        [OCutils.get_class_i(x_val, y_val, classDict[key]) for key in class_out],
        OCutils.train_transforms
    )

    if args.action == "train":
        logger = logging.getLogger('logging_lambda')
        logger_start(logger, args)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cifar_loader_tc = data.DataLoader(cifar_train_target, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers)
        oe_sampler = data.RandomSampler(cifar_train_oe, replacement=True, num_samples=len(cifar_train_target))
        cifar_loader_oe = data.DataLoader(cifar_train_oe, batch_size=args.batch_size, shuffle=False,
                                          sampler=oe_sampler, num_workers=args.num_workers)
        train(model, cifar_loader_tc, cifar_loader_oe, optimizer, args)
    else:
        device = torch.device("cpu")
        cifar_loader = data.DataLoader(cifar_train_target, batch_size=args.batch_size, num_workers=args.num_workers)
        cifar_loader_val_target = data.DataLoader(cifar_val_target, batch_size=args.batch_size,
                                                  num_workers=args.num_workers)
        if len(args.class_out) > 1:
            out_sampler = data.RandomSampler(cifar_val_out, replacement=True, num_samples=len(cifar_val_target))
            cifar_loader_val_outliers = data.DataLoader(cifar_val_out, batch_size=args.batch_size,
                                                        sampler=out_sampler, num_workers=args.num_workers)
        else:
            cifar_loader_val_outliers = data.DataLoader(cifar_val_out, batch_size=args.batch_size,
                                                        num_workers=args.num_workers)

        model.load_state_dict(
            torch.load('../export_models/' + folder + sub_folder_target + sub_folder_oe + args.model_test + '.pt',
                       map_location=device))
        model.to(device)
        # get features from train data for train the SVM
        features, labels = OCutils.get_features(model, cifar_loader)
        # features_val, labels_val = get_features(model, cifar_loader_val, relabel=True)
        features_val_target, labels_var_target = OCutils.get_features(model, cifar_loader_val_target)
        features_val_out, labels_val_out = OCutils.get_features(model, cifar_loader_val_outliers,
                                                                relabel_as_outliers=True)
        # test with three classes:
        # features_val_out, labels_val_out = get_features(model, cifar_loader_val_outliers, label_plus_one=True)
        labels_val = torch.cat((labels_var_target, labels_val_out), 0)
        features_val = torch.cat((features_val_target, features_val_out), 0)
        # concat target and outlier data for validation
        # tSNE visualisation
        if tnes_vis:
            tsne = TSNE(n_components=2).fit_transform(features_val)
            tx = tsne[:, 0]
            ty = tsne[:, 1]
            tx = Aux.scale_to_01_range(tx)
            ty = Aux.scale_to_01_range(ty)
            plot_tsne(tx, ty, labels_val, args.class_target, args.class_out, title=args.model_test, save=True,
                      title_extra=title_plot,
                      folder=folder + sub_folder_target + sub_folder_oe)
        features2D = features.reshape(features.shape[0], -1)
        features2D_val = features_val.reshape(features_val.shape[0], -1)
        # takes features for training the SVM and features for validation
        OCSVM(features2D, features2D_val, args.model_test, labels_val)
