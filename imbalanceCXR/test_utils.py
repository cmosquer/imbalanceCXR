import os, sys
sys.path.insert(0,"..")

import pickle
from os.path import join
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score,  brier_score_loss, log_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc as sklearnAUC
from imbalanceCXR.utils import tqdm

try:
    from imbalanceCXR.calibration import logregCal, PAV
    CALIBRATION_AVAILABLE = True
except Exception as e:
    print(e, "Couldnt import logregCal, wont apply calibration")
    CALIBRATION_AVAILABLE = False


def getCalibrationErrors(labels, probs,
                         bin_upper_bounds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                         save_details_pathology=None):
    num_bins = len(bin_upper_bounds)
    bin_indices = np.digitize(probs, bin_upper_bounds)
    counts = np.bincount(bin_indices, minlength=num_bins)
    nonzero = counts != 0

    accuracies_sklearn, confidences_sklearn = calibration_curve(labels, probs, n_bins=num_bins)
    if save_details_pathology:
        with open(save_details_pathology, 'wb') as f:
            pickle.dump([accuracies_sklearn, confidences_sklearn, counts], f)
    calibration_errors = accuracies_sklearn - confidences_sklearn
    weighting = counts / float(len(probs.flatten()))
    weighted_calibration_errors = np.abs(calibration_errors) * weighting[nonzero]

    ece = np.sum(weighted_calibration_errors)
    mce = np.max(calibration_errors)
    return ece, mce


def getCalibrationMetrics(labels, probs, save_details_pathology=None):
    positive_labels = labels[labels == 1]
    Npos = len(positive_labels)
    positive_preds = probs[labels == 1]

    negative_labels = labels[labels == 0]
    negative_preds = probs[labels == 0]

    # Calibration errors
    try:
        ece, mce = getCalibrationErrors(labels, probs, save_details_pathology=save_details_pathology)
        ecePos, mcePos = getCalibrationErrors(positive_labels, positive_preds)
        eceNeg, mceNeg = getCalibrationErrors(negative_labels, negative_preds)
    except:
        ece, mce, ecePos, mcePos, eceNeg, mceNeg = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Brier scores
    assert len(positive_labels) + len(negative_labels) == len(labels)

    brierPos = brier_score_loss(positive_labels, positive_preds)
    brierNeg = brier_score_loss(negative_labels, negative_preds)
    brier = brier_score_loss(labels, probs)

    # Negative log likelihood
    nll = log_loss(labels, probs)

    return Npos, ece, mce, ecePos, mcePos, eceNeg, mceNeg, brier, brierPos, brierNeg, nll


def getMetrics(y_true, y_pred, metrics_results, YI_thresholds_roc, save_details_pathology=None, costs_thr=None):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    youden_index_thres = thr[np.argmax(tpr - fpr)]
    YI_thresholds_roc.append(youden_index_thres)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_precision_recall = sklearnAUC(recall, precision)

    Npos, ece, mce, ecePos, \
    mcePos, eceNeg, mceNeg, \
    brier, brierPos, brierNeg, nllSklearn = getCalibrationMetrics(y_true,
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

    return metrics_results, YI_thresholds_roc


def valid_epoch(name, epoch, model, device, data_loader, criterions, priors=None,
                limit=None, cfg=None, dataset_name='', save_preds=False):

    if cfg is not None:
        save_preds = cfg.save_preds
    model.eval()

    n_count = {}
    pathology_outputs = {}
    pathology_targets = {}
    pathology_outputs_sigmoid = {}
    pathology_outputs_sigmoid_calibrated = {}
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

            for pathology in range(len(pathology_targets)):
                pathology_output = outputs[:, pathology]
                pathology_target = targets[:, pathology]
                mask = ~torch.isnan(pathology_target)  # We use the samples where this pathology is positive
                pathology_output = pathology_output[mask]
                pathology_target = pathology_target[mask]
                pathology_output_sigmoid = torch.sigmoid(pathology_output).detach().cpu().numpy()

                pathology_outputs_sigmoid[pathology].append(pathology_output_sigmoid)
                pathology_outputs[pathology].append(pathology_output.detach().cpu().numpy())
                pathology_targets[pathology].append(pathology_target.detach().cpu().numpy())

                if len(pathology_target) > 0:
                    for loss_function, criterion in criterions.items():
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

        for loss_function, losses in avg_loss_results.items():
            txt += f'\n{loss_function}:'
            for pathology in range(len(pathology_targets)):
                avg_loss_results[loss_function][pathology] /= n_count[pathology]

                txt += f'{pathology}: {avg_loss_results[loss_function][pathology].item()}'
        t.set_description(f'Epoch {epoch + 1} - {txt}')

        # Once we infered all batches and sum their losses, we unify predictions to average loss per pathology
        if name == 'test':
            with open(join(cfg.output_dir, name, f'{dataset_name}-calibrator_parameters.pkl'), 'rb') as f:
                calibration_parameters = pickle.load(f)
        else:
            calibration_parameters = {}
        for pathology in range(len(pathology_targets)):
            pathology_outputs[pathology] = np.concatenate(pathology_outputs[pathology])
            pathology_outputs_sigmoid[pathology] = np.concatenate(pathology_outputs_sigmoid[pathology])
            pathology_targets[pathology] = np.concatenate(pathology_targets[pathology])
            if CALIBRATION_AVAILABLE:

                # Calibration with dca_plda package
                epsilon = 1e-100
                positive_posteriors = pathology_outputs_sigmoid[pathology]
                negative_posteriors = 1 - pathology_outputs_sigmoid[pathology]
                targets = pathology_targets[pathology]
                train_positive_prior = priors['train']['priors_pos'][pathology]
                train_negative_prior = priors['train']['priors_neg'][pathology]
                LLR = np.log((positive_posteriors + epsilon) / (negative_posteriors + epsilon)) - np.log(
                    (train_positive_prior + epsilon) / (train_negative_prior + epsilon))

                tar = LLR[targets == 1]
                non = LLR[targets == 0]
                print('Len tar {} Len non {}'.format(len(tar), len(non)))
                ptar = priors['valid']['priors_pos'][pathology]
                theta = np.log(cost_ratio * (1 - ptar) / ptar)
                ptar_hat = 1 / (1 + np.exp(theta))
                if name=='test':
                    # Apply linear calibrator that was fit with validation set
                    a = calibration_parameters[pathology]['a']
                    b = calibration_parameters[pathology]['b']
                    k = calibration_parameters[pathology]['k']

                    #Fit PAV algorithm as reference of perfectly calibrated version of the model
                    sc = np.concatenate((tar, non))
                    la = np.zeros_like(sc, dtype=int)
                    la[:len(tar)] = 1.0
                    calibration_parameters[pathology]["pav"] = PAV(sc, la)
                else:
                    #Fit a linear calibrator to the validation set
                    a, b = logregCal(tar, non, ptar_hat, return_params=True)
                    k = -np.log((1 - ptar) / ptar)
                    print('a {:.2f} b {:.2f} k {:.2f}'.format(a,b,k))
                    calibration_parameters[pathology] = {'a': a, 'b': b, 'k': k}
                pathology_outputs_sigmoid_calibrated[pathology] = 1 / (1 + np.exp(-(a * LLR + b) + k))

        if name!='test':
            with open(join(cfg.output_dir, name, f'{dataset_name}-calibrator_parameters.pkl'), 'wb') as f:
                pickle.dump(calibration_parameters, f)

        if save_preds:
            os.makedirs(join(cfg.output_dir, name), exist_ok=True)
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
                   'AUC-ROC', 'AUC-PR',
                   'f1score-0.5', 'f1score-costsTh',
                   'accuracy-0.5', 'accuracy-costsTh',
                   'nllSklearn']

        metrics_results = {}
        for metric in metrics:
            metrics_results[metric] = []

        YI_thresholds_roc = []
        for pathology in range(len(pathology_targets)):
            if len(np.unique(pathology_targets[pathology])) > 1:
                y_true, y_pred = np.array(pathology_targets[pathology], dtype=np.int64), pathology_outputs_sigmoid[
                    pathology]

                metrics_results, YI_thresholds_roc = getMetrics(y_true, y_pred,
                                                                metrics_results,
                                                                YI_thresholds_roc)
            else:
                for metric in metrics:
                    metrics_results[metric].append(np.nan)
            for loss_function, criterion in criterions.items():
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
    for metric, mean in metrics_means.items():
        print_string += f' Avg {metric}={mean:4.4f}  '
    print(print_string)
    if CALIBRATION_AVAILABLE:

        metrics_results_calibrated = {}
        for metric in metrics:
            metrics_results_calibrated[metric] = []
        thresholds_roc_calibrated = []
        for pathology in range(len(pathology_targets)):
            if len(np.unique(pathology_targets[pathology])) > 1:
                y_true, y_pred = np.array(pathology_targets[pathology], dtype=np.int64), \
                                 pathology_outputs_sigmoid_calibrated[pathology]
                ptar = priors['valid']['priors_pos']
                Tau_bayes = cost_ratio * (1 - ptar) / ptar

                th_posteriors = Tau_bayes / (1 + Tau_bayes)

                print('\nCOSTS TH: ', th_posteriors)
                metrics_results_calibrated, thresholds_roc_calibrated = getMetrics(y_true, y_pred,
                                                                                   metrics_results_calibrated,
                                                                                   thresholds_roc_calibrated,
                                                                                   costs_thr=th_posteriors,
                                                                                   )

            else:
                for metric in metrics:
                    metrics_results_calibrated[metric].append(np.nan)

        # Add calibrated dictionary to metrics_results dictionary
        for oldkey in metrics:
            metrics_results_calibrated[oldkey + '_calibrated'] = metrics_results_calibrated.pop(oldkey)

        metrics_results.update(metrics_results_calibrated)


        # TODO: add calibration with PAV to find minimum Brier as reference for calibration performance

        if name=='test':
            for pathology in range(len(pathology_targets)):
                pav = calibration_parameters[pathology]['pav']
                llrs, ntar, nnon = pav.llrs()
                print(pathology,llrs)
                print(ntar,nnon)
                """
                for p in np.atleast_1d(ptar):
    
                    logitPost = llrs + logit(p)
    
                    Ctar, Cnon = softplus(-logitPost), softplus(logitPost)
                    min_cllr = p*(Ctar[ntar!=0] @ ntar[ntar!=0]) / ntar.sum() +  (1-p)*(Cnon[nnon!=0] @ nnon[nnon!=0]) / nnon.sum()  
                    min_cllr /= -p*np.log(p) - (1-p)*np.log(1-p)
                """

    return metrics_means['AUC-ROC'], metrics_means[
        'AUC-PR'], metrics_results, thresholds, pathology_outputs, pathology_targets

