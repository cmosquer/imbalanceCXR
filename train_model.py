# coding: utf-8

import os,sys, pickle
sys.path.insert(0,".")
import numpy as np

import torch
import torchvision, torchvision.transforms
from imbalanceCXR.configure_datasets import parseDatasets
import random
from imbalanceCXR.train_utils import train
from imbalanceCXR.test_utils import valid_epoch
from imbalanceCXR.utils import getModel
import torchxrayvision as xrv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')


parser.add_argument('--output_dir', type=str, default="./output/", help='Path where outputs will be saved')
parser.add_argument('--dataset', type=str, default="chex", help='Chest X-ray Datasets to use')
parser.add_argument('--model', type=str, default="densenet121", help='Deep Learning arquitecture to train')
parser.add_argument('--cuda', type=bool, default=False, help='Use GPU')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='Train and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
parser.add_argument('--shuffle', type=bool, default=True, help='If True, data CSVs are shuffled')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--threads', type=int, default=4, help='Number of threads')
parser.add_argument('--loss_function', type=str, default='NLL', help='Loss function for training. Default is negative log likelihood (NLL).\
                                                                        "WNLL" for weighted NLL. "focal" for Focal Loss.')
parser.add_argument('--data_aug', type=bool, default=True, help='Whether to apply image transformations')
parser.add_argument('--data_aug_rot', type=int, default=45, help='If data_aug is True, degrees of rotation')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='If data_aug is True, proportion of translation')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='If data_aug is True, proportion of scale')

parser.add_argument('--save_all_models', type=bool, default=False, help='If True, all epochs are saved. If False, save only best epochs according to selection_metric')
parser.add_argument('--save_preds',type=bool,default=False,help='If True, save the targets and preds of the validation set')
parser.add_argument('--selection_metric',type=str,default='roc',help=' "roc" o "pr". Which AUC to use for model selection and checkpoint saving')

parser.add_argument('--n_seeds',type=bool,default=False,help='If True, "seed" integer is considered as the number of seeds to use. '
                                                             'Range from 0 to "seed"-1 will be used as seed. The whole configured proccess will be performed "seed" times')
parser.add_argument('--seed', type=int, default=0, help='If n_seeds is True, seed determines the number of times the experiment is repeated. Otherwise it determines the specific spliting seed to use for the experiment')

parser.add_argument('--only_test', type=str, default=False, help='Skip training')
parser.add_argument('--only_train', type=str, default=False, help='Skip testing')

cfg = parser.parse_args()
print(cfg)

assert cfg.loss_function in ['NLL','WNLL','focal']
assert cfg.selection_metric in ['roc','pr']

data_aug = None
if cfg.data_aug:
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(cfg.data_aug_rot, 
                                            translate=(cfg.data_aug_trans, cfg.data_aug_trans), 
                                            scale=(1.0-cfg.data_aug_scale, 1.0+cfg.data_aug_scale)),
        torchvision.transforms.ToTensor()
    ])
    print(data_aug)

transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

datas, datas_names = parseDatasets(cfg.dataset,transforms,data_aug)

print("dataset names", datas_names)

for d in datas:
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d)

if cfg.n_seeds:
    seed_list = range(cfg.seed)
else:
    seed_list = [cfg.seed]

for _seed in seed_list:

    cfg.seed = _seed
    print('------------STARTING SEED {}-------------'.format(cfg.seed))
    #cut out training sets
    train_datas = []
    test_datas = []
    for i, dataset in enumerate(datas):
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        torch.manual_seed(cfg.seed)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))

        #disable data aug
        test_dataset.data_aug = None

        #fix labels
        train_dataset.labels = dataset.labels[train_dataset.indices]
        test_dataset.labels = dataset.labels[test_dataset.indices]

        train_dataset.csv = dataset.csv.iloc[train_dataset.indices]
        test_dataset.csv = dataset.csv.iloc[test_dataset.indices]

        train_dataset.pathologies = dataset.pathologies
        test_dataset.pathologies = dataset.pathologies

        train_datas.append(train_dataset)
        test_datas.append(test_dataset)

    if len(datas) == 0:
        raise Exception("no dataset")
    elif len(datas) == 1:
        train_dataset = train_datas[0]
        test_dataset = test_datas[0]
    else:
        print("merge datasets")
        train_dataset = xrv.datasets.Merge_Dataset(train_datas)
        test_dataset = xrv.datasets.Merge_Dataset(test_datas)


    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("train_dataset.labels.shape", train_dataset.labels.shape)
    print("test_dataset.labels.shape", test_dataset.labels.shape)
    print('positives test', np.nansum(test_dataset.labels, axis=0))

    print("train_dataset",train_dataset)
    print("test_dataset",test_dataset)

    # create models
    num_classes = train_dataset.labels.shape[1]
    model = getModel(cfg.model,num_classes)
    
    device = 'cuda' if cfg.cuda else 'cpu'
    dataset_name = "{}-{}-seed{}-{}".format(cfg.dataset, cfg.model, cfg.seed, cfg.loss_function)
    os.makedirs(cfg.output_dir + '/valid', exist_ok=True)
    if not cfg.only_test:

        train(model, train_dataset, dataset_name, cfg)
        print("Done training")

    if not cfg.only_train:
        print("Loading best weights")
        weights_file = cfg.output_dir, f'/{dataset_name}-best_{cfg.selection_metric}.pt'
        model.load_state_dict(torch.load(weights_file))
        model.to(device)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.test_batch_size,
                                                  shuffle=cfg.shuffle,
                                                  num_workers=cfg.threads, pin_memory=cfg.cuda)
        print("Starting test")

        with open(os.path.join(cfg.output_dir, f'{dataset_name}-priors.pkl'), "rb") as f:
            priors_dict = pickle.load(f)
        os.makedirs(cfg.output_dir+'/test', exist_ok=True)

        test_auc, test_performance_metrics, test_thresholds, _, _ = valid_epoch(name='test',
                                                                                 epoch='test',
                                                                                 model=model,
                                                                                 device=device,
                                                                                 data_loader=test_loader,
                                                                                 criterions=torch.nn.BCEWithLogitsLoss(),
                                                                                 priors=True, dataset_name=dataset_name,
                                                                                 cfg=cfg)

        with open(cfg.output_dir, '/test/', f'{dataset_name}-test-performance-metrics.pkl', 'wb') as f:
            pickle.dump(test_performance_metrics, f)

        with open(cfg.output_dir, '/test/', f'{dataset_name}-test-thresholds.pkl', "wb") as f:
            pickle.dump(test_thresholds, f)






