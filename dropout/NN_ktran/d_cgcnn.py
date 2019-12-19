#!/usr/bin/env python
# coding: utf-8


# # Initialization
# Importing modules
import numpy as np
import pandas as pd
import os, sys, pickle, torch
from torch.optim import Adam
import skorch.callbacks.base
from skorch import callbacks  # needs skorch >= 0.4  
from skorch import NeuralNetRegressor
from skorch.dataset import CVSplit
from cgcnn.dropoutmodel import CrystalGraphConvNet
from cgcnn.data import collate_pool, MergeDataset
sys.path.append(os.path.expanduser("~/cgcnn"))


# Define the dropout on all forward passes
dropout = float(sys.argv[1])
dropout_bool = (dropout != False)
dropout_percent = int(dropout * 100) # in percent form
print('Raw given dropout', dropout)


# Automatically search for an NVIDIA GPU and use it. If not, then use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = 'cpu'
    print('Using CPU')


# Load all of our preprocessed and split data from our cache
model_name = 'CGCNN standalone'

with open('../../preprocessing/sdt/gasdb/feature_dimensions.pkl', 'rb') as file_handle:
    orig_atom_fea_len, nbr_fea_len = pickle.load(file_handle)

with open('../../preprocessing/splits_gasdb.pkl', 'rb') as file_handle:
    splits = pickle.load(file_handle)

docs_train, docs_val, docs_test = splits['docs_train'], splits['docs_val'], splits['docs_test']
sdts_train, sdts_val, sdts_test = splits['sdts_train'], splits['sdts_val'], splits['sdts_test']
targets_train, targets_val, targets_test = splits['targets_train'], splits['targets_val'], splits['targets_test']


# Initialize the CGCNN `net` class
# Callback to checkpoint parameters every time there is a new best for validation loss
cp = callbacks.Checkpoint(monitor='valid_loss_best', fn_prefix='./histories/d%i_valid_best_' % dropout_percent)

# Callback to load the checkpoint with the best validation loss at the end of training
class train_end_load_best_valid_loss(skorch.callbacks.base.Callback):
    def on_train_end(self, net, X, y):
        net.load_params('./histories/d%i_valid_best_params.pt' % dropout_percent)
load_best_valid_loss = train_end_load_best_valid_loss()

# Callback to set the learning rate dynamically
LR_schedule = callbacks.lr_scheduler.LRScheduler('MultiStepLR', milestones=[100], gamma=0.1)

net = NeuralNetRegressor(
    CrystalGraphConvNet,
    module__orig_atom_fea_len=orig_atom_fea_len,
    module__nbr_fea_len=nbr_fea_len,
    batch_size=214,
    module__classification=False,
    lr=0.0056,
    max_epochs=150,
    module__atom_fea_len=46,
    module__h_fea_len=83,
    module__n_conv=8,
    module__n_h=4,
    optimizer=Adam,
    iterator_train__pin_memory=True,
    iterator_train__num_workers=0,
    iterator_train__collate_fn=collate_pool,
    iterator_train__shuffle=True,
    iterator_valid__pin_memory=True,
    iterator_valid__num_workers=0,
    iterator_valid__collate_fn=collate_pool,
    iterator_valid__shuffle=False,
    device=device,
    criterion=torch.nn.L1Loss,
    dataset=MergeDataset,
    dropout_bool_=dropout_bool,
    dropout_weight_=dropout,
    callbacks=[cp, load_best_valid_loss, LR_schedule]
)


# # Training

# We can train a new model...
f_history   = './histories/valid_best_d%i_history.json' % dropout_percent
f_optimizer = './histories/valid_best_d%i_optimizer.pt' % dropout_percent
f_params    = './histories/valid_best_d%i_params.pt' % dropout_percent

"""
# We can save time by using previously cached parameters
if (os.path.exists(f_history) and 
    os.path.exists(f_optimizer) and 
    os.path.exists(f_params) and
    retrain == False):
    net.initialize()
    net.load_params(f_history=f_history,
                    f_optimizer=f_optimizer, 
                    f_params=f_params)
else:
    net.initialize()
    net.fit(sdts_train_, targets_train_)
"""

net.initialize()
net.fit(sdts_train, targets_train)

net.initialize()
net.load_params(f_history=f_history,
                f_optimizer=f_optimizer, 
                f_params=f_params)

# # Assess performance
from tqdm import tqdm_notebook

# Calculate the error metrics
predy = np.array([net.predict(sdts_test) for _ in tqdm_notebook(range(100))])
ymean = predy.mean(axis=1)
standard_errors = predy.std(axis=1)
residuals = ymean - targets_test.reshape(-1)


# Calculate scalar metrics
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             median_absolute_error)
mae   = mean_absolute_error(targets_test, ymean)
rmse  = np.sqrt(mean_squared_error(targets_test, ymean))
r2    = r2_score(targets_test, ymean)
mdae  = median_absolute_error(targets_test, ymean)
marpd = np.abs(2 * residuals /
               (np.abs(ymean) + np.abs(targets_test.reshape(-1)))
               ).mean() * 100
corr  = np.corrcoef(targets_test.reshape(-1), ymean)[0, 1]

# Save as pickle to be plotted with in the same graph as others
with open('histories/single_cgcnn_d%i.pkl' % dropout_percent, 'wb') as saveplot:
    pickle.dump((predy, targets_test, mae, rmse, r2, mdae, marpd, corr), saveplot)