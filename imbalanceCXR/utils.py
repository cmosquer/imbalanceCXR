import os, sys
sys.path.insert(0,"..")

import torchvision.models as torch_mod
import numpy as np
import torch
import torchxrayvision as xrv
from focal_loss.focal_loss import FocalLoss
from tqdm import tqdm as tqdm_base
from matplotlib import pyplot as plt


def plotBrierMetrics(means, stds, sorted_pathologies, to_plot_metrics=None,
                               plot_type='combined',labels=None):


    styles_dict = {
        'brierPos': {'label': 'Brier+', 'color': 'mediumseagreen', 'lw': '1', 'ls': '--'},
        'brierNeg': {'label': 'Brier-', 'color': 'lightcoral', 'lw': '1', 'ls': '-.'},
        'brier': {'label': 'Brier', 'color': 'dodgerblue', 'lw': '1'},
        'balancedBrier': {'label': 'Balanced Brier', 'color': 'deepskyblue', 'lw': '1'},
    }

    if to_plot_metrics is None:
        to_plot_metrics = styles_dict.keys()

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    x = 2.5 * np.arange(len(sorted_pathologies))
    width = 0.4  # the width of the bars
    aucs_count = 0
    acc_count = 0
    bars_count = 0
    metric_count = 0
    for i, metric in enumerate(to_plot_metrics):
        if plot_type == 'combined':
            combined = True
            alpha_bars = 0.5
            if metric == 'brier' or metric == 'balancedBrier':
                plot_type = 'points'
            else:
                plot_type = 'bars'
        else:
            combined = False
            alpha_bars = 0.8

        if plot_type == 'points':
            width = 0.3
            horiz = x + metric_count * width + width
            err_color = styles_dict[metric]['color']

            if metric == 'brier' or metric == 'balancedBrier':
                if metric == 'brier':
                    marker_ = 'o'
                else:
                    marker = 'v'
                horiz = x + acc_count * width
                fs = 14
                lw = 2
                ls = '--'
                style = 'normal'

                points = ax.scatter(horiz, means[metric], label=styles_dict[metric]['label'],

                                    marker=marker_,
                                    color=styles_dict[metric]['color'])
                ax.plot(horiz, means[metric],
                        ls=ls, lw=lw,
                        color=styles_dict[metric]['color'])

            else:
                horiz = x + acc_count * width
                fs = 12
                lw = 1
                ls = '-.'
                style = 'italic'
                if metric == 'brierPos':
                    marker_ = 'x'
                if metric == 'brierNeg':
                    marker_ = 'v'

                points = ax.scatter(horiz, means[metric], label=styles_dict[metric]['label'],
                                    marker=marker_,
                                    color=styles_dict[metric]['color'])

            for x_, y_, std in zip(horiz, means[metric], stds[metric]):
                #    ax.text(x_,y_+std+0.02,
                #            f'{y_:.2f}',horizontalalignment='center',style=style,
                #            color='blue',#styles_dict[metric]['color'],
                #            fontsize=fs)
                ax.vlines(x_, ymin=y_ - std, ymax=y_ + std,
                          color=styles_dict[metric]['color'], alpha=0.3,
                          lw=1)
        if plot_type == 'bars':
            width = 0.3
            horiz = x - width / 2 + metric_count * width
            err_color = 'lightgray'  # styles_dict[metric]['color']
            ax.bar(horiz, means[metric], width, alpha=alpha_bars, edgecolor='gray',
                   label=styles_dict[metric]['label'],
                   color=styles_dict[metric]['color'],
                   yerr=stds[metric], ecolor=err_color)
            metric_count += 1

        if combined:
            plot_type = 'combined'

    ax.set_ylim((0, 1))
    ax.set_ylabel('Brier metrics', fontsize=14)
    # ax.set_xticks(x + metric_count*width/4)
    ax.set_xticks(x)
    if labels is not None:
        ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize='large', markerscale=2)
    fig.tight_layout()
    return fig

def plotDiscriminationMetrics(means, stds, sorted_pathologies, to_plot_metrics=None,
                               plot_type='combined',labels=None):
    styles_dict = {
        'AUC-ROC': {'label': 'AUC-ROC', 'color': 'darkgreen', 'lw': '2'},
        'AUC-PR': {'label': 'AUC-PR', 'color': 'firebrick', 'lw': '2'},
        'recall': {'label': 'Recall', 'color': 'orange', 'lw': '2'},
        'precision': {'label': 'Precision', 'color': 'purple', 'lw': '2'},
        'specificity': {'label': 'Specificity', 'color': 'deepskyblue', 'lw': '2'},
    }
    if to_plot_metrics is None:
        to_plot_metrics = styles_dict.keys()
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))
    x = 2.5 * np.arange(len(sorted_pathologies))
    width = 0.4  # the width of the bars
    metric_count = 0
    for i, metric in enumerate(to_plot_metrics):

        if 'AUC' in metric:
            marker_ = 'o'
            fs = 14
            lw = 2
            ls = '--'
            style = 'normal'
        else:
            fs = 12
            lw = 1
            ls = '-.'
            style = 'italic'
            if metric == 'recall':
                marker_ = 'x'
            if metric == 'precision':
                marker_ = 'v'
            if metric == 'specificity':
                marker_ = '*'
        if plot_type == 'combined':
            combined = True
            alpha_bars = 0.5
            if 'AUC' in metric:
                plot_type = 'points'
            else:
                plot_type = 'bars'
        else:
            combined = False
            alpha_bars = 0.8
        if plot_type == 'points':
            width = 0.3
            horiz = x + metric_count * width + width
            err_color = styles_dict[metric]['color']
            points = ax.scatter(horiz, means[metric], label=styles_dict[metric]['label'],
                                marker=marker_,
                                color=styles_dict[metric]['color'])
            ax.plot(horiz, means[metric], alpha=0.3,
                    ls=ls, lw=lw,
                    color=styles_dict[metric]['color'])
            for x_, y_, std in zip(horiz, means[metric], stds[metric]):
                # ax.text(x_,y_+std+0.02,
                #        f'{y_:.2f}',horizontalalignment='center',style=style,
                #        color='blue',#styles_dict[metric]['color'],
                #        fontsize=fs)
                ax.vlines(x_, ymin=y_ - std, ymax=y_ + std,
                          color=styles_dict[metric]['color'], alpha=0.2,
                          lw=0.5)

        if plot_type == 'bars':
            width = 0.3
            horiz = x + metric_count * width
            err_color = 'lightgray'  # styles_dict[metric]['color']
            ax.bar(horiz, means[metric], width, alpha=alpha_bars, edgecolor='gray',
                   label=styles_dict[metric]['label'],
                   color=styles_dict[metric]['color'],
                   yerr=stds[metric], ecolor=err_color)
            metric_count += 1

        if combined:
            plot_type = 'combined'
    ax.set_ylabel('Discrimination metrics', fontsize=14)
    ax.set_xticks(x + (metric_count / 2) * width)
    # ax.set_xticks(x + metric_count*width+width)
    if labels is not None:
        ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim((0, 1.1))
    ax.legend()
    fig.tight_layout()
    return fig

def getModel(modelName, num_classes):
    if "densenet" in modelName:
        model = xrv.models.DenseNet(num_classes=num_classes, in_channels=1,
                                    **xrv.models.get_densenet_params(modelName))
    elif "resnet101" in modelName:
        model = torch_mod.resnet101(num_classes=num_classes, pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif "shufflenet_v2_x2_0" in modelName:
        model = torch_mod.shufflenet_v2_x2_0(num_classes=num_classes, pretrained=False)
        #patch for single channel
        model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
    else:
        raise Exception("no model")
    return model


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)



def getCriterions(loader):
    n_pathologys = loader.dataset[0]["lab"].shape[0]

    N_total_validos = np.count_nonzero(~np.isnan(loader.dataset.labels), axis=0)
    N_total_validos_pos = np.count_nonzero(loader.dataset.labels==1, axis=0)
    N_total_validos_neg = np.count_nonzero(loader.dataset.labels==0, axis=0)

    priors_pos = N_total_validos_pos / N_total_validos
    priors_neg = N_total_validos_neg / N_total_validos
    pos_weights = torch.Tensor(N_total_validos_neg / (N_total_validos_pos + 1e-7))

    criterions_dict = {}
    for criterion in ['NLL','WNLL','focal']:
        if 'NLL' in criterion:
            value = [torch.nn.BCEWithLogitsLoss()] * n_pathologys
        if criterion=='WNLL':
            for pathology in range(n_pathologys):
                value[pathology] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights[pathology])
        if criterion=='focal':
            value = [FocalLoss(alpha=2, gamma=5)] * n_pathologys
        criterions_dict[criterion] = value

    priors_dict = {'n_total': N_total_validos,
                    'n_pos': N_total_validos_pos,
                    'n_neg': N_total_validos_neg,
                    'priors_pos': priors_pos,
                    'priors_neg': priors_neg}

    return criterions_dict, priors_dict



