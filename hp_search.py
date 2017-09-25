import json
import os
import os.path

from data import VOCAB_SIZE
from models import Seq2Seq
from train import train_iters

RESULTS_PATH = 'results/'


def experiment(train, val, batch_size, hidden_size, lr, iters, print_iters):
    print 'Building model...'
    model = Seq2Seq(VOCAB_SIZE, hidden_size).cuda()

    print 'Training...'
    train_losses, val_losses = train_iters(model, train, val, batch_size, lr, iters, print_iters)

    print 'Saving...'
    result = {'lr': lr, 'hidden_size': hidden_size, 'batch_size': batch_size,
              'train_losses': train_losses, 'val_losses': val_losses}
    result = json.dumps(result, indent=2, sort_keys=True)

    file_name = 'lr_{}_hidden_size_{}_batch_size_{}.json'.format(lr, hidden_size, batch_size)
    file_path = os.path.join(RESULTS_PATH, file_name)
    with open(file_path, 'w') as f:
        f.write(result)
