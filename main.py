# -*- coding: utf-8 -*-
import os
import pickle
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from time import time
import datetime, random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from GraphTranszuizong import *
from utilszuizong import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./datasets/')
parser.add_argument("--feature_path", type=str, default='./feature/')
parser.add_argument("--task", type=str, default='PRO') 
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_id", type=str, default=None)

args = parser.parse_args()

seed = args.seed
root = args.feature_path + 'input_ESM_dis/'
Dataset_Path = args.dataset_path
Feature_Path = args.feature_path
run_id = args.run_id
task = args.task

Seed_everything(seed=seed)


if task == "Metal":
    model_class = MetalSite
else:
    model_class = GraphTrans


train_df = pd.read_csv(Dataset_Path + task +  '_train.csv')
test_df = pd.read_csv(Dataset_Path + task + '_test.csv')


if args.train:
    ID_list = list(set(train_df['ID']))
elif args.test:
    ID_list = list(set(test_df['ID']))


all_protein_data = {}
for pdb_id in ID_list:
    if task == "Metal":
        all_protein_data[pdb_id]=torch.load(root+f"{pdb_id}_X.tensor"),torch.load(root+f"{pdb_id}_node_feature.tensor"),torch.load(root+f"{pdb_id}_mask.tensor") # ,torch.load(root+f"{pdb_id}_label_mask.tensor"),torch.load(root+f"{pdb_id}_label.tensor")
    else:
        all_protein_data[pdb_id]=torch.load(root+f"{pdb_id}_X.tensor"),torch.load(root+f"{pdb_id}_node_feature.tensor"),torch.load(root+f"{pdb_id}_mask.tensor") # ,torch.load(root+f"{pdb_id}_label.tensor")


train_size = {"CA":820, "MG":914, "MN":259,"FE":163,"CO":70,"ZN":930,"CU":95}  
num_samples = train_size[task] * 9

nn_config = {
    'node_features': 2560 + 14, # esmTrans + 14D
    'edge_features': 16,
    'hidden_dim': 64,
    'num_encoder_layers': 4,
    'k_neighbors': 30,
    'augment_eps': 0.1,
    'dropout': 0.4,
    'id_name':'ID',
    'obj_max': 1,
    'epochs': 30,
    'patience': 5,
    'batch_size': 16,
    'num_samples': num_samples,
    'folds': 5,
    'seed': seed,
    'remark': task + ' binding site prediction'
}


if args.train:
    NN_train_and_predict(train_df, None, all_protein_data, model_class, nn_config, logit = True, run_id = run_id, args=args)
elif args.test:
    NN_train_and_predict(None, test_df, all_protein_data, model_class, nn_config, logit = True, run_id = run_id, args=args)