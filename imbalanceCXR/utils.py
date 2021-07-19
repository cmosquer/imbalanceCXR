import os
import pickle
import pprint
import random
from glob import glob
from os.path import exists, join
import torchvision.models as torch_mod
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score,  brier_score_loss, log_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc as sklearnAUC
import torchxrayvision as xrv
from focal_loss.focal_loss import FocalLoss
from dca_plda.calibration import logregCal
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


def getCalibrationErrors(labels,probs,
                         bin_upper_bounds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                         save_details_pathology=None):
    num_bins = len(bin_upper_bounds)
    bin_indices = np.digitize(probs, bin_upper_bounds)
    counts = np.bincount(bin_indices, minlength=num_bins)
    nonzero = counts != 0

    accuracies_sklearn, confidences_sklearn = calibration_curve(labels, probs, n_bins=num_bins)
    if save_details_pathology:
        with open(save_details_pathology, 'wb') as f:
            pickle.dump([accuracies_sklearn,confidences_sklearn,counts],f)
    calibration_errors = accuracies_sklearn - confidences_sklearn
    weighting = counts / float(len(probs.flatten()))
    weighted_calibration_errors = np.abs(calibration_errors) * weighting[nonzero]

    ece = np.sum(weighted_calibration_errors)
    mce = np.max(calibration_errors)
    return ece,mce


def getCalibrationMetrics(labels,probs, save_details_pathology=None):
    positive_labels = labels[labels == 1]
    Npos = len(positive_labels)
    positive_preds = probs[labels == 1]

    negative_labels = labels[labels == 0]
    negative_preds = probs[labels == 0]

    #Calibration errors
    try:
        ece, mce = getCalibrationErrors(labels,probs,save_details_pathology=save_details_pathology)
        ecePos, mcePos = getCalibrationErrors(positive_labels,positive_preds)
        eceNeg, mceNeg = getCalibrationErrors(negative_labels,negative_preds)
    except:
        ece, mce, ecePos, mcePos, eceNeg, mceNeg = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    #Brier scores
    assert len(positive_labels) + len(negative_labels) == len(labels)

    brierPos = brier_score_loss(positive_labels, positive_preds)
    brierNeg = brier_score_loss(negative_labels, negative_preds)
    brier = brier_score_loss(labels, probs)

    #Negative log likelihood
    nll = log_loss(labels,probs)

    return Npos, ece, mce, ecePos, mcePos, eceNeg, mceNeg, brier, brierPos, brierNeg, nll


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


def getMetrics(y_true,y_pred,save_details_pathology,metrics_results,YI_thresholds_roc,costs_thr=None):

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    youden_index_thres = thr[np.argmax(tpr - fpr)]
    YI_thresholds_roc.append(youden_index_thres)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_precision_recall = sklearnAUC(recall, precision)

    Npos, ece, mce, ecePos, \
    mcePos, eceNeg, mceNeg, \
    brier, brierPos, brierNeg,nllSklearn = getCalibrationMetrics(y_true,
                                                           y_pred,
                                                           save_details_pathology=save_details_pathology,
                                                           )

    metrics_results['AUC-ROC'].append(roc_auc_score(y_true, y_pred))
    metrics_results['f1score-0.5'].append(f1_score(y_true, y_pred > 0.5))
    metrics_results['accuracy-0.5'].append(accuracy_score(y_true, y_pred > 0.5))
    metrics_results['AUC-PR'].append(auc_precision_recall)
    metrics_results['Npos'].append(Npos)
    metrics_results['ECE'].append(ece)
    metrics_results['MCE'].append(mce)
    metrics_results['ECE+'].append(ecePos)
    metrics_results['MCE+'].append(mcePos)
    metrics_results['ECE-'].append(eceNeg)
    metrics_results['MCE-'].append(mceNeg)
    metrics_results['brier'].append(brier)
    metrics_results['brier+'].append(brierPos)
    metrics_results['brier-'].append(brierNeg)
    metrics_results['nllSklearn'].append(nllSklearn)

    if costs_thr is not None:
        metrics_results['f1score-costsTh'].append(f1_score(y_true, y_pred > costs_thr))
        metrics_results['accuracy-costsTh'].append(accuracy_score(y_true, y_pred > costs_thr))

    return metrics_results,YI_thresholds_roc


def train(model, dataset, dataset_name, cfg):
    print("Our config:")
    pprint.pprint(cfg)        

    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    print(cfg.output_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir,exist_ok=True)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    torch.manual_seed(cfg.seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size],
                                                                 generator=torch.Generator().manual_seed(42))
    
    #disable data aug
    valid_dataset.data_aug = None
    
    # fix labels
    train_dataset.labels = dataset.labels[train_dataset.indices]
    valid_dataset.labels = dataset.labels[valid_dataset.indices]

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.batch_size,
                                                   shuffle=cfg.shuffle,
                                                   num_workers=cfg.threads,
                                                   pin_memory=cfg.cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)

    scheduler = StepLR(optim, step_size=40, gamma=0.1)
    criterions_train, priors_train = getCriterions(train_loader)
    criterions_valid, priors_valid = getCriterions(valid_loader)

    priors_dict = {'train': priors_train,
                   'valid': priors_valid}

    with open(join(cfg.output_dir, f'{dataset_name}-priors.pkl'), "wb") as f:
        pickle.dump(priors_dict, f)
        
    start_epoch = 0
    best_metric_roc = 0.
    best_metric_pr = 0.
    weights_for_best_validauc = None

    metrics = []
    weights_files = glob(join(cfg.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())
        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric_roc = metrics[-1]['best_metric_roc']
        best_metric_pr = metrics[-1]['best_metric_pr']

    model.to(device)
    
    for epoch in range(start_epoch, cfg.num_epochs):

        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               scheduler=scheduler,
                               criterions=criterions_train)
        
        aucroc_valid, aucpr_valid, current_performance_metrics,thresholds,_,_ = valid_epoch(
                                     name='valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterions=criterions_valid,
                                     priors=priors_dict,
                                     dataset_name=dataset_name,
                                     )

        if os.path.exists(join(cfg.output_dir, f'{dataset_name}-performance-metrics.pkl')):
            with open(join(cfg.output_dir, f'{dataset_name}-performance-metrics.pkl'), 'rb') as f:
                performance_metrics = pickle.load(f)
            performance_metrics.append(current_performance_metrics)
        else: #First epoch
            performance_metrics = [current_performance_metrics]
        with open(join(cfg.output_dir, f'{dataset_name}-performance-metrics.pkl'), 'wb') as f:
            pickle.dump(performance_metrics, f)

        if np.mean(aucroc_valid) > best_metric_roc:
            best_metric_roc = np.mean(aucroc_valid)
            print('new best roc ',best_metric_roc)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best_roc.pt'))
            with open(join(cfg.output_dir, f'{dataset_name}-best-thresholds_roc.pkl'),"wb") as f:
                pickle.dump(thresholds,f)
        if np.mean(aucpr_valid) > best_metric_pr:
            best_metric_pr = np.mean(aucpr_valid)
            print('new best pr ',best_metric_pr)

            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best_pr.pt'))

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validaucroc": aucroc_valid,
            "validaucpr": aucpr_valid,
            'best_metric_roc': best_metric_roc,
            'best_metric_pr': best_metric_pr
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        if cfg.save_all_models:
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

    return metrics, best_metric_roc, weights_for_best_validauc


def train_epoch(cfg, epoch, model, device,  optimizer,
                train_loader, criterions, scheduler=None, limit=None):
    model.train()
    avg_loss = []
    t = tqdm(train_loader)
    training_criterion = criterions[cfg.loss_function]
    for batch_idx, samples in enumerate(t):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break

        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)

        outputs = model(images)
        
        loss = torch.zeros(1).to(device).float()
        for pathology in range(targets.shape[1]):
            pathology_output = outputs[:,pathology]
            pathology_target = targets[:,pathology]
            mask = ~torch.isnan(pathology_target)
            pathology_output = pathology_output[mask]
            pathology_target = pathology_target[mask]
            if len(pathology_target) > 0:
                pathology_loss = training_criterion[pathology](pathology_output.float(), pathology_target.float())
                loss += pathology_loss
                
        loss = loss.sum()

        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')


        optimizer.step()

    if scheduler:
        scheduler.step()

    return np.mean(avg_loss)

def valid_epoch(name, epoch, model, device, data_loader, criterions, priors=None,
                     limit=None, cfg=None, dataset_name='',save_preds=False,
                     calibration_parameters=None, ):
    model.eval()

    n_count = {}
    pathology_outputs={}
    pathology_targets={}
    pathology_outputs_sigmoid={}
    pathology_outputs_sigmoid_calibrated={}
    avg_loss_results = dict.fromkeys(criterions.keys())
    for loss_function in avg_loss_results.keys():
        avg_loss_results[loss_function] = {}
        for pathology in range(data_loader.dataset[0]["lab"].shape[0]):
            avg_loss_results[loss_function][pathology] = torch.zeros(1).to(device).double()
    for pathology in range(data_loader.dataset[0]["lab"].shape[0]):
        pathology_outputs[pathology] = []
        pathology_targets[pathology] = []
        n_count[pathology] = 0
        pathology_outputs_sigmoid[pathology] = []
        pathology_outputs_sigmoid_calibrated[pathology] = []

    cost_ratio = 1 / 1  # Cost of false positives over cost of false negatives. TODO: Make it configurable for each pathology

    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break

            images = samples["img"].to(device)
            targets = samples["lab"].to(device)
            outputs = model(images)

            for pathology in range(targets.shape[1]): 
                pathology_output = outputs[:,pathology]
                pathology_target = targets[:,pathology]
                mask = ~torch.isnan(pathology_target) #We use the samples where this pathology is positive
                pathology_output = pathology_output[mask]
                pathology_target = pathology_target[mask]
                pathology_output_sigmoid = torch.sigmoid(pathology_output).detach().cpu().numpy()


                pathology_outputs_sigmoid[pathology].append(pathology_output_sigmoid)
                pathology_outputs[pathology].append(pathology_output.detach().cpu().numpy())
                pathology_targets[pathology].append(pathology_target.detach().cpu().numpy())

                if len(pathology_target) > 0:
                    for loss_function,criterion in criterions.items():
                        criterion_pathology = criterion[pathology]
                        batch_loss_pathology = criterion_pathology(pathology_output.double(), pathology_target.double())
                        avg_loss_results[loss_function][pathology] += batch_loss_pathology

                    n_count[pathology] += len(samples)

            del images
            del outputs
            del samples
            del targets

        txt = ''
        print('ncounts: ', n_count)
        print('avg_loss_results: ', avg_loss_results)
        for loss_function, losses in avg_loss_results.items():
            txt += f'\n{loss_function}:'
            for pathology in range(targets.shape[1]):
                avg_loss_results[loss_function][pathology] /= n_count[pathology]

                txt += f'{pathology}: {avg_loss_results[loss_function][pathology].item()}'
        t.set_description(f'Epoch {epoch + 1} - {txt}')

        #Once we infered all batches and sum their losses, we unify predictions to average loss per pathology

        for pathology in range(len(pathology_targets)):
            pathology_outputs[pathology] = np.concatenate(pathology_outputs[pathology])
            pathology_outputs_sigmoid[pathology] = np.concatenate(pathology_outputs_sigmoid[pathology])
            pathology_targets[pathology] = np.concatenate(pathology_targets[pathology])

            # Calibration with dca_plda package
            epsilon=1e-100
            positive_posteriors = pathology_outputs_sigmoid[pathology]
            negative_posteriors = 1-pathology_outputs_sigmoid[pathology]
            train_positive_prior = priors['train']['priors_pos'][pathology]
            train_negative_prior = priors['train']['priors_neg'][pathology]
            LLR = np.log((positive_posteriors+epsilon)/(negative_posteriors+epsilon)) - np.log((train_positive_prior+epsilon)/(train_negative_prior+epsilon))

            tar = LLR[targets == 1]
            non = LLR[targets == 0]
            ptar = priors['valid']['priors_pos'][pathology]
            theta = np.log(cost_ratio* (1 - ptar) / ptar)
            ptar_hat = 1 / (1 + np.exp(theta))
            a, b = logregCal(tar, non, ptar_hat, return_params=True)
            k = -np.log((1 - ptar) / ptar)

            pathology_outputs_sigmoid_calibrated[pathology] = 1 / (1 + np.exp(-(a*LLR + b) + k))

        if save_preds:
            os.makedirs(join(cfg.output_dir, name),exist_ok=True)
            results_dict = {'targets': pathology_targets,
                        'probas': pathology_outputs_sigmoid,
                        'logits': pathology_outputs,
                        'calibrated_probas': pathology_outputs_sigmoid_calibrated}
            with open(join(cfg.output_dir, name, f'{dataset_name}-predictions.pkl'), 'wb') as f:
                pickle.dump(results_dict, f)

        metrics = ['Npos',
                    'ECE', 'MCE',
                    'ECE+', 'MCE+',
                    'ECE-', 'MCE-',
                    'brier', 'brier+', 'brier-',
                    'AUC-ROC','AUC-PR',
                    'f1score-0.5','f1score-costsTh',
                    'accuracy-0.5','accuracy-costsTh',
                   'nllSklearn']

        metrics_results = {}
        for metric in metrics:
            metrics_results[metric] = []

        YI_thresholds_roc = []
        for pathology in range(len(pathology_targets)):
            if len(np.unique(pathology_targets[pathology]))> 1:
                y_true, y_pred = np.array(pathology_targets[pathology],dtype=np.int64), pathology_outputs_sigmoid[pathology]

                metrics_results, YI_thresholds_roc = getMetrics(y_true, y_pred,
                                                                   metrics_results,
                                                                   YI_thresholds_roc)
            else:
                for metric in metrics:
                    metrics_results[metric].append(np.nan)
            for loss_function,criterion in criterions.items():
                metrics_results[loss_function] = avg_loss_results[loss_function][pathology]
    print('NLLs ', metrics_results['NLL'])
    print('Weighted NLLs ', metrics_results['weightedNLL'])

    metrics_means = {}
    for metric in metrics:
        metrics_results[metric] = np.asarray(metrics_results[metric])
        metrics_means[metric] = np.mean(metrics_results[metric][~np.isnan(metrics_results[metric])])
    thresholds = np.array(YI_thresholds_roc)

    if 'test' not in name:
        print(f'Epoch {epoch + 1} - {name}')
    print_string = ''
    for metric,mean in metrics_means.items():
        print_string += f' Avg {metric}={mean:4.4f}  '
    print(print_string)

    metrics_results_calibrated = {}
    for metric in metrics:
        metrics_results_calibrated[metric] = []
    thresholds_roc_calibrated = []
    for pathology in range(len(pathology_targets)):
        if len(np.unique(pathology_targets[pathology]))> 1:
            y_true,y_pred = np.array(pathology_targets[pathology],dtype=np.int64), pathology_outputs_sigmoid_calibrated[pathology]
            ptar = priors['valid']['priors_pos']
            Tau_bayes = cost_ratio * (1 - ptar) / ptar

            th_posteriors = Tau_bayes / (1 + Tau_bayes)

            print('\nCOSTS TH: ',th_posteriors)
            metrics_results_calibrated,thresholds_roc_calibrated = getMetrics(y_true,y_pred,None,
                                                                              metrics_results_calibrated,
                                                                              thresholds_roc_calibrated,
                                                                              costs_thr=th_posteriors,
                                                                              )

        else:
            for metric in metrics:
                metrics_results_calibrated[metric].append(np.nan)

    #Add calibrated dictionary to metrics_results dictionary
    for oldkey in metrics:
        metrics_results_calibrated[oldkey + '_calibrated'] = metrics_results_calibrated.pop(oldkey)

    metrics_results.update(metrics_results_calibrated)

    return metrics_means['AUC-ROC'], metrics_means['AUC-PR'], metrics_results, thresholds, pathology_outputs, pathology_targets

