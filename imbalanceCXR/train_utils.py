import os, sys
sys.path.insert(0,"..")

import pickle
import pprint
import random
from glob import glob
from os.path import exists, join
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from imbalanceCXR.utils import getCriterions, tqdm
from imbalanceCXR.test_utils import valid_epoch


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
        os.makedirs(cfg.output_dir, exist_ok=True)

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

    # disable data aug
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

        aucroc_valid, aucpr_valid, current_performance_metrics, thresholds, _, _ = valid_epoch(
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
        else:  # First epoch
            performance_metrics = [current_performance_metrics]
        with open(join(cfg.output_dir, f'{dataset_name}-performance-metrics.pkl'), 'wb') as f:
            pickle.dump(performance_metrics, f)

        if np.mean(aucroc_valid) > best_metric_roc:
            best_metric_roc = np.mean(aucroc_valid)
            print('new best roc ', best_metric_roc)
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best_roc.pt'))
            with open(join(cfg.output_dir, f'{dataset_name}-best-thresholds_roc.pkl'), "wb") as f:
                pickle.dump(thresholds, f)
        if np.mean(aucpr_valid) > best_metric_pr:
            best_metric_pr = np.mean(aucpr_valid)
            print('new best pr ', best_metric_pr)

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


def train_epoch(cfg, epoch, model, device, optimizer,
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
            pathology_output = outputs[:, pathology]
            pathology_target = targets[:, pathology]
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
