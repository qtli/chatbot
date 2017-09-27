import json
import os

import numpy as np
import visdom

from hp_search import RESULTS_PATH

vis = visdom.Visdom()
for name in os.listdir(RESULTS_PATH):
    path = os.path.join(RESULTS_PATH, name)
    if os.path.isfile(path):
        with open(path, 'r') as f:
            experiment = json.load(f)

        title = 'lr: {}, hidden size: {}, batch size: {}'.format(experiment['lr'], experiment['hidden_size'],
                                                                 experiment['batch_size'])

        train_losses = np.array(experiment['train_losses'])
        val_losses = np.array(experiment['val_losses'])
        iters = np.arange(1, train_losses.shape[0] + 1)

        vis.line(Y=np.column_stack((train_losses, val_losses)),
                 X=np.column_stack((iters, iters)),
                 opts={
                     'legend': ['train', 'val'],
                     'title': title
                 })
