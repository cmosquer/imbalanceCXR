
# imbalanceCXR

#### This repository is based on the library [torchrayvision](https://github.com/mlmed/torchxrayvision) by the [Machine Learning and Medicine Lab](https://mlmed.org/w/)

We focused on adding metrics to evaluate overall performance in imbalanced data, including both the discrimination and calibration aspects
This repository can be used to reproduce the experiments described in the article "Understanding the impact of class imbalance on the performance of chest x-ray image classifiers"

### Getting started

- Clone this repository
- Install the requirements: 
``pip install -r requirements.txt``
- Follow instructions to install the [DCA_PLDA](https://github.com/luferrer/DCA-PLDA) package. 
- Copy the script calibration.py to the imbalanceCXR folder
``cp DCA-PLDA/dca_plda/calibration.py imbalanceCXR/imbalanceCXR``
- This repository uses public chest x-ray datasets. Modify the script configure_datasets.py to indicate the directories where the datasets are downloaded.

### Train and test models with multiple splitting seeds

``python train_model.py --output_dir="/home/myuser/CXRoutput/" --dataset=chex --n_seeds=True --seed=5 --save_preds=True --cuda=True`` 

### Plot results

``python plot_results.py --output_dir="/home/myuser/CXRoutput/" --figures_dir="/home/myuser/CXRoutput/figures" --dataset=chex --n_seeds=5`` 
