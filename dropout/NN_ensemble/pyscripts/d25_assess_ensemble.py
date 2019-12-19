#!/usr/bin/env python
# coding: utf-8

# Initialization
import torch
import pickle
import numpy as np
from sklearn.model_selection import KFold
from torch.optim import Adam
import skorch.callbacks.base
from skorch.callbacks import Checkpoint  # needs skorch >= 0.4
from skorch.callbacks.lr_scheduler import LRScheduler
from skorch import NeuralNetRegressor
from cgcnn.dropoutmodel25 import CrystalGraphConvNet
from cgcnn.data import collate_pool, MergeDataset


# Find and use the appropriate GPU/CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = 'cpu'
    print('Using CPU')


# Load the data split from our Jupyter notebook cache
model_name = 'CGCNN ensemble'

with open('../../preprocessing/sdt/gasdb/feature_dimensions.pkl', 'rb') as file_handle:
    orig_atom_fea_len, nbr_fea_len = pickle.load(file_handle)

with open('../../preprocessing/splits_gasdb.pkl', 'rb') as file_handle:
    splits = pickle.load(file_handle)

docs_train, docs_val, docs_test = splits['docs_train'], splits['docs_val'], splits['docs_test']
sdts_train, sdts_val, sdts_test = splits['sdts_train'], splits['sdts_val'], splits['sdts_test']
targets_train, targets_val, targets_test = splits['targets_train'], splits['targets_val'], splits['targets_test']

class train_end_load_best_valid_loss(skorch.callbacks.base.Callback):
    def on_train_end(self, net, X, y):
        net.load_params('./histories/%i_d25_valid_best_params.pt' % k)

nets = []


# Fold the CV data and train
k_folder = KFold(n_splits=5)
for k, (indices_train, _) in enumerate(k_folder.split(sdts_train)):
    stds_train_ = [sdts_train[index] for index in indices_train]
    targets_train_ = np.array([targets_train[index] for index in indices_train])

    # Define various callbacks and checkpointers for this network
    LR_schedule = LRScheduler('MultiStepLR', milestones=[75], gamma=0.1)
    cp = Checkpoint(monitor='valid_loss_best', fn_prefix='./histories/%i_d25_valid_best_' % k)
    load_best_valid_loss = train_end_load_best_valid_loss()

    # Train this fold's network
    net = NeuralNetRegressor(
        CrystalGraphConvNet,
        module__orig_atom_fea_len=orig_atom_fea_len,
        module__nbr_fea_len=nbr_fea_len,
        batch_size=214,
        module__classification=False,
        lr=0.0056,
        max_epochs=100,
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
        callbacks=[cp, load_best_valid_loss, LR_schedule]
    )
    net.initialize()
    net.fit(stds_train_, targets_train_)
    nets.append(net)


# Load existing neural net configurations (optional)
k_folder = KFold(n_splits=5)
for k, (indices_train, _) in enumerate(k_folder.split(sdts_train)):
    stds_train_ = [sdts_train[index] for index in indices_train]
    targets_train_ = np.array([targets_train[index] for index in indices_train])

    # Define various callbacks and checkpointers for this network
    LR_schedule = LRScheduler('MultiStepLR', milestones=[75], gamma=0.1)
    cp = Checkpoint(monitor='valid_loss_best', fn_prefix='./histories/%i_d25_valid_best_' % k)
    load_best_valid_loss = train_end_load_best_valid_loss()

    # Train this fold's network
    net = NeuralNetRegressor(
        CrystalGraphConvNet,
        module__orig_atom_fea_len=orig_atom_fea_len,
        module__nbr_fea_len=nbr_fea_len,
        batch_size=214,
        module__classification=False,
        lr=0.0056,
        max_epochs=100,
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
        callbacks=[cp, load_best_valid_loss, LR_schedule]
    )
    net.initialize()
    net.load_params(f_history='./histories/%i_d25_valid_best_history.json' % k,
                    f_optimizer= './histories/%i_d25_valid_best_optimizer.pt' % k, 
                    f_params='./histories/%i_d25_valid_best_params.pt' % k)
    nets.append(net)

    
# Ensembling
class Ensemble:
    def __init__(self, networks):
        self.networks = networks

    def predict(self, features, iters):
        for i, net in enumerate(self.networks):
            for _ in range(iters):
                prediction = net.predict(features)
                try:
                    predictions = np.hstack((predictions, prediction))
                except NameError:
                    predictions = prediction
            print('Net %i finished training.' % i)

        return predictions

iters = 100 # Use net.predict() 100 times for every use of net.fit()


# Make the predictions and calculate metrics in vector form
ensemble = Ensemble(nets)
predy = ensemble.predict(sdts_test, iters)
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
with open('histories/assess_ensemble_plots_d25.pkl', 'wb') as saveplot:
    pickle.dump((predy, targets_test, mae, rmse, r2, mdae, marpd, corr), 
                saveplot)
