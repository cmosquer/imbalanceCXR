import pickle
import os,sys
sys.path.insert(0,"..")

import numpy as np

from sklearn.metrics import auc,brier_score_loss, roc_curve, roc_auc_score, confusion_matrix,precision_recall_curve

from sklearn.metrics import auc as sklearnAUC
from utils import plotDiscriminationMetrics,plotBrierMetrics

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="./output/", help='Path where train outputs were saved')
parser.add_argument('--figures_dir', type=str, default="./output/", help='Path to save figures')
parser.add_argument('--dataset', type=str, default="chex", help='Chest X-ray Datasets to use. Currently available for nih or chex')
parser.add_argument('--trained_model', type=str, default="densenet121", help='Deep Learning arquitecture trained')
parser.add_argument('--n_seeds',type=int,default=5,help='Plot seeds from 0 to n-1')


pathologies  = {'default':['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia','Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'],
                'nih':["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema", "Fibrosis","Effusion", "Pneumonia", "Pleural_Thickening","Cardiomegaly", "Nodule", "Mass", "Hernia"],
                'chex':["Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia", "Atelectasis","Pneumothorax","Effusion","Fracture",
                            #"Pleural Other", "Support Devices", 'No Finding'
                   ]}
cfg = parser.parse_args()
dataset = cfg.dataset
n_seeds = cfg.n_seeds
plot_type = 'combined'  # 'bars','points' or 'combined'

files_name = '{}-{}-seed'.format(dataset,cfg.trained_model)

os.makedirs(cfg.figures_dir+'/png',exist_ok=True)
os.makedirs(cfg.figures_dir+'/svg',exist_ok=True)

priors = np.array([None] * n_seeds)
for file in os.listdir(cfg.output_dir):
    if 'priors' in file and dataset in file:
        seed = int(file.split('seed')[-1][0])
        if seed < n_seeds:
            print(file)
            with open(os.path.join(cfg.output_dir, file), 'rb') as f:
                priors[seed] = pickle.load(f)

sorted_pathologies = [x[1] for x in sorted(zip(priors[0]['valid']['n_pos'], pathologies['default']), reverse=True) if
                      x[1] in pathologies[dataset]]

results_test = np.array([None] * n_seeds)
results_valid = np.array([None] * n_seeds)

test_metrics = np.array([None] * n_seeds)
valid_metrics = np.array([None] * n_seeds)


testDir = cfg.output_dir + '/test/'
validDir = cfg.output_dir + '/valid/'

thresholds = {}

for file in os.listdir(testDir):
    if files_name in file:
        if 'performance-metrics' in file:
            seed = int(file.split('seed')[-1][0])
            if seed < n_seeds:
                with open(os.path.join(testDir, file), 'rb') as f:
                    test_metrics[seed] = pickle.load(f)
        if 'predict' in file:
            seed = int(file.split('seed')[-1][0])
            if seed < n_seeds:
                with open(os.path.join(testDir, file), 'rb') as f:
                    results_test[seed] = pickle.load(f)

for file in os.listdir(validDir):
    if files_name in file:
        if 'performance-metrics' in file:
            seed = int(file.split('seed')[-1][0])
            if seed < n_seeds:
                with open(os.path.join(validDir, file), 'rb') as f:
                    valid_metrics[seed] = pickle.load(f)
        if 'predict' in file:
            seed = int(file.split('seed')[-1][0])
            if seed < n_seeds:
                # print('found valid preds for seed ',seed)
                with open(os.path.join(validDir, file), 'rb') as f:
                    results_valid[seed] = pickle.load(f)

mean_n_pos = [np.mean(
    [priors[seed]['test']['n_pos'][pathology] / priors[seed]['test']['n_total'][pathology] for seed in range(n_seeds)])
              for pathology in range(len(pathologies['default']))]
mean_n_pos_int = [np.mean([priors[seed]['test']['n_pos'][pathology] for seed in range(n_seeds)]) for pathology in
                  range(len(pathologies['default']))]

## DISCRIMINATION PLOTS

to_plot_metrics_discrimination = [
                    'AUC-ROC',
                    'AUC-PR',
                    'recall',
                    'precision',
                    'specificity',
                   ]

means = dict.fromkeys(to_plot_metrics_discrimination)
stds = dict.fromkeys(to_plot_metrics_discrimination)

for metric in to_plot_metrics_discrimination:
    means[metric] = np.zeros(len(sorted_pathologies))
    stds[metric] = np.zeros(len(sorted_pathologies))
labels = []
for pathology_id, pathology_name in enumerate(sorted_pathologies):
    if pathology_name in pathologies[dataset]:
        current = pathologies['default'].index(pathology_name)
        maxf1threhsold = np.array([None] * n_seeds)

        if len(pathology_name.split(' ')) > 1:
            pathname = pathology_name.split(' ')[0][:7] + '\n' + pathology_name.split(' ')[1][:7] + '.'
        else:
            pathname = pathology_name[:7]

        labels.append("{} \n({:.2f}%)".format(pathname,
                                              100 * mean_n_pos[current]))
        for seed in range(n_seeds):
            y_valid = (results_valid[seed]['targets'][current], results_valid[seed]['probas'][current])
            if np.sum(y[1]) == 0:
                print('all zeros')
            # Get optimal threshold
            precision_valid, recall_valid, ths_prec_recall_valid = precision_recall_curve(y_valid[0], y_valid[1])
            if len(recall_valid[recall_valid != 0]) == 0 and len(precision_valid[precision_valid != 0]) == 0:
                print('quilombo')
                f1_scores_valid = np.zeros(len(precision_valid))

            f1_scores_valid = 2 * recall_valid * precision_valid / (recall_valid + precision_valid + 1e-15)

            maxf1threhsold[seed] = ths_prec_recall_valid[np.argmax(f1_scores_valid)]
            maxrecall_valid = recall_valid[np.argmax(f1_scores_valid)]
            maxprecision_valid= precision_valid[np.argmax(f1_scores_valid)]
            print(f'{pathology_name}: Best F1 for {seed} had recall={maxrecall_valid} and precision={maxprecision_valid}')
            print('Max recall: {}. Max precision: {}. Max F1 score: {}'.format(recall_valid.max(),
                                                                               precision_valid.max(),
                                                                               f1_scores_valid.max()))
        thresholds[pathology_name] = maxf1threhsold

        for metric in to_plot_metrics_discrimination:
            metric_array = np.zeros(n_seeds)
            for seed in range(n_seeds):
                y = (results_test[seed]['targets'][current], results_test[seed]['probas'][current])

                if metric == 'AUC-ROC':
                    this_metric = roc_auc_score(y[0], y[1])

                if metric == 'AUC-PR':
                    precision, recall, ths_prec_recall = precision_recall_curve(y[0], y[1])
                    this_metric = sklearnAUC(recall, precision)

                if 'AUC' not in metric:
                    y_binaria = (y[0], y[1] > maxf1threhsold[seed])

                    c = confusion_matrix(y_binaria[0], y_binaria[1])
                    tp = c[1][1]
                    fp = c[0][1]
                    tn = c[0][0]
                    fn = c[1][0]

                    if metric == 'precision':

                        if tp == 0:
                            this_metric = 0
                        else:
                            this_metric = tp / (fp + tp)
                    if metric == 'recall':
                        if tp == 0:
                            this_metric = 0
                        else:
                            this_metric = tp / (tp + fn)
                    if metric == 'specificity':
                        if tn == 0:
                            this_metric = 0
                        else:
                            this_metric = tn / (tn + fp)
                metric_array[seed] = this_metric
            means[metric][pathology_id] = np.mean(metric_array)
            stds[metric][pathology_id] = np.std(metric_array)

discrimination_fig = plotDiscriminationMetrics(means, stds, sorted_pathologies, to_plot_metrics=to_plot_metrics_discrimination,
                               plot_type=plot_type,labels=labels)

discrimination_fig.savefig(cfg.figures_dir+f'/png/discrimination_{dataset}_{plot_type}.png',format='png',dpi=500)
discrimination_fig.savefig(cfg.figures_dir+f'/svg/discrimination_{dataset}_{plot_type}.svg',format='svg')

to_plot_metrics_brier = ['brier',
                   'balancedBrier',
                   'brierPos',
                   'brierNeg']
means = dict.fromkeys(to_plot_metrics_brier)
stds = dict.fromkeys(to_plot_metrics_brier)

for metric in to_plot_metrics_brier:
    means[metric] = np.zeros(len(sorted_pathologies))
    stds[metric] = np.zeros(len(sorted_pathologies))
labels = []
for pathology_id, pathology_name in enumerate(sorted_pathologies):
    if pathology_name in pathologies[dataset]:
        current = pathologies['default'].index(pathology_name)

        if len(pathology_name.split(' ')) > 1:
            pathname = pathology_name.split(' ')[0][:7] + '\n' + pathology_name.split(' ')[1][:7] + '.'
        else:
            pathname = pathology_name[:8]

        labels.append("{} \n({:.2f}%)".format(pathname,
                                              100 * mean_n_pos[current]))

        for metric in to_plot_metrics_brier:
            metric_array = np.zeros(n_seeds)
            for seed in range(n_seeds):

                y = (results_test[seed]['targets'][current], results_test[seed]['probas'][current])

                if metric == 'brier':
                    this_metric = brier_score_loss(y[0], y[1])
                if metric == 'brierPos':
                    this_metric = brier_score_loss(y[0][y[0] == 1], y[1][y[0] == 1])
                    # print(this_metric,test_metrics[seed][metric][current])
                if metric == 'brierNeg':
                    this_metric = brier_score_loss(y[0][y[0] == 0], y[1][y[0] == 0])
                if metric == 'balancedBrier':
                    brier_neg = brier_score_loss(y[0][y[0] == 0], y[1][y[0] == 0])
                    brier_pos = brier_score_loss(y[0][y[0] == 1], y[1][y[0] == 1])
                    this_metric = brier_neg + brier_pos
                metric_array[seed] = this_metric
            # print(metric_array,np.mean(metric_array))
            means[metric][pathology_id] = np.mean(metric_array)
            stds[metric][pathology_id] = np.std(metric_array)


briers_fig = plotBrierMetrics(means, stds, sorted_pathologies, to_plot_metrics=to_plot_metrics_brier,
                               plot_type=plot_type,labels=labels)

briers_fig.savefig(cfg.figures_dir+f'/png/briers_{dataset}_{plot_type}.png',format='png',dpi=500)
briers_fig.savefig(cfg.figures_dir+f'/svg/briers_{dataset}_{plot_type}.svg',format='svg')