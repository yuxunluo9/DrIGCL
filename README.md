# DrIGCL

This repository contains an implementation of DrIGCL model based on the paper "A computational 
framework for predicting novel drug indications using graph convolutional network with contrastive 
learning" by Yuxun Luo, Wenyu Shan, Li Peng, Lingyun Luo, Pingjian Ding, and Wei Liang.

## Overview

This project implements the DrIGCL model proposed in the paper for a computational framework for 
predicting novel drug indications using graph convolutional network with contrastive learning. 
DrIGCL is proposed for drug indication prediction, which utilizes graph convolutional networks and 
contrastive learning. DrIGCL incorporates drug structure, disease comorbidities, and known drug 
indications to extract representations of drugs and diseases. By combining contrastive and 
classification losses, DrIGCL predicts drug indications effectively.

## Requirements

* Python 3.8.18
* pytorch=1.12.1
* pytorch-metric-learning=2.4.1
* dgl=1.1.1.cu113
* scikit-learn=1.3.0
* pandas=2.0.3
* numpy=1.24.3

## Installation

```angular2html
conda env create -f environment.yml
conda activate DrIGCL
python main_hov.py
```

## Model Architecture

The DrIGCL model architecture follows the details provided in the paper. The architecture of 
DrIGCL comprises five components: graph construction, graph convolutional network submodule, 
multi-head attention submodule, contrastive optimization, and drug-disease association prediction 
optimization. The implementation supports customizable hyperparameters such as the number of 
hidden units, the number of layers, and dropout rate.

## Usage

To train and evaluate the DrIGCL model with 20 runs, use the following command:

```angular2html
python main_hov.py --runs 20
```

To train and evaluate the DrIGCL's variant, without contrastive learning:

```angular2html
python main_hov.py --loss_weight 0.0
```

To train and evaluate the DrIGCL's variant, without drug similarity:

```angular2html
python main_hov.py --drug_ablation True
```

To train and evaluate the DrIGCL's variant, without disease comorbidity:

```angular2html
python main_hov.py --disease_ablation True
```

## Results

The model achieves an AUC of 0.923, Hits@1 of 0.365, Hits@3 of 0.453, and 
Hits@10 of 0.586 on the test set after 100 epochs of training. However, the results can vary 
due to the randomness of the train/val/test split.