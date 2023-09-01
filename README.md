# Multimodal Variational Contrastive Learning for Few-Shot Classification (MVC)

PyTorch implementation for the paper: Multimodal Variational Contrastive Learning for Few-Shot Classification

## Dependencies
* python 3.6.5
* numpy 1.16.0
* torch 1.8.0
* tqdm 4.57.0
* scipy 1.5.4
* torchvision 0.9.0

## Overview
The effectiveness of metric-based few-shot learning methods heavily relies on the discriminative ability of the prototypes and feature embeddings of queries. However, using instance-level unimodal prototypes often falls short in capturing the essence of various categories. To this end, we propose a multimodal variational contrastive learning framework that aims to enhance prototype representativeness and refine the discrimination of query features by acquiring distribution-level representations. To elaborate, our approach commences by training a variational auto-encoder through supervised contrastive learning in both the visual and semantic spaces. The trained model is employed to augment the support set by repetitive sampling features from the learned semantic distributions and generate pseudo-semantics for queries to achieve information balance across samples in both the support and query sets. Furthermore, we establish a multimodal instance-to-distribution model that learns to transform instance-level multimodal features into distribution-level representations via variational inference, facilitating robust metric. Empirical experiments conducted across several benchmarks consistently demonstrates the superiority of our method in terms of classification accuracy and robustness.
![Image text](https://github.com/pmhDL/MVC/blob/main/Architecture/Architecture.png)

## Download the Datasets
* [miniImageNet](https://drive.google.com/file/d/1g4wOa0FpWalffXJMN2IZw0K2TM2uxzbk/view) 
* [tieredImageNet](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view?usp=sharing)
* [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)
* [CUB](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)
* [glove word embedding](https://nlp.stanford.edu/projects/glove/)

## Running Experiments
If you want to train the models from scratch, please run the run_pre.py first to pretrain the backbone. Then specify the path of the pretrained checkpoints to "./checkpoints/[dataname]"
* Run pretrain phase:
```bash
python run_pre.py
```
* Run few-shot training and test phases:
```bash
python run_mvc.py
```
## LISENCE
* All materials are made available under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0) license. You can find details at: https://creativecommons.org/licenses/by-nc/4.0/legalcode

* The license gives permission for academic use only.
