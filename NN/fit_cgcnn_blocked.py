'''
This job takes longer than 4 hours, which is the time limit for Jupyter-GPU on
NERSC. So we use this Python script to do the fitting in a SLURM job instead.
'''

import pickle
import numpy as np
import torch
from torch.optim import Adam
from skorch import callbacks  # needs skorch >= 0.4
from skorch import NeuralNetRegressor
from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import collate_pool, MergeDataset


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = 'cpu'
    print('Using CPU')

with open('../preprocessing/sdt/gasdb/feature_dimensions.pkl', 'rb') as file_handle:
    orig_atom_fea_len, nbr_fea_len = pickle.load(file_handle)

with open('../preprocessing/splits_gasdb.pkl', 'rb') as file_handle:
    splits = pickle.load(file_handle)

docs_train = splits['docs_train']
docs_val = splits['docs_val']
sdts_train = splits['sdts_train']
sdts_val = splits['sdts_val']
targets_train = splits['targets_train']
targets_val = splits['targets_val']

# Where we put the intermediate results for this notebook
prefix = 'gasdb_blocked/'

# Define all the adsorbates
adsorbates = list({doc['adsorbate'] for doc in docs_val})
adsorbates.sort()

# Initialize all of the objects we'll use to train
nets = {}
cps = {}
best_finders = {}
lr_schedulers = {}

# Block by adsorbate
for ads in adsorbates:

    # Callback to checkpoint parameters every time there is a new best for validation loss
    cp = callbacks.Checkpoint(monitor='valid_loss_best', fn_prefix=prefix + 'valid_best_%s_' % ads)

    # Callback to load the checkpoint with the best validation loss at the end of training
    class train_end_load_best_valid_loss(callbacks.base.Callback):
        def on_train_end(self, net, X, y):
            net.load_params(prefix + 'valid_best_%s_params.pt' % ads)
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
        callbacks=[cp, load_best_valid_loss, LR_schedule]
    )

    # Assign everything to their respective dictionaries
    nets[ads] = net
    cps[ads] = cp
    best_finders[ads] = load_best_valid_loss
    lr_schedulers[ads] = LR_schedule

# Block the data
for ads, net in nets.items():
    _sdts_train = []
    _targets_train = []
    for doc, sdt, target in zip(docs_train, sdts_train, targets_train):
        if doc['adsorbate'] == ads:
            _sdts_train.append(sdt)
            _targets_train.append(target)
    _targets_train = np.array(_targets_train)

    # Fit
    net.initialize()
    net.fit(_sdts_train, _targets_train)
