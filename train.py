import time

import torch
from torch.optim import SGD
from torch.autograd import Variable

from data import TRAIN_FILE_NAME
from data import VAL_FILE_NAME
from data import pad_seqs
from dataset import Dataset
from models import Seq2Seq
from models import init_hidden
from utils import plot_loss

VOCAB_SIZE = 20000
HIDDEN_SIZE = 256
LR = 1e-1
BATCH_SIZE = 64

ITERS = 1000
PRINT_ITERS = 100


def get_loss(model, batch, inference_only=False):
    answer_lens = [len(example['answer']) for example in batch]

    questions = pad_seqs([example['question'] for example in batch])
    answers = pad_seqs([example['answer'] for example in batch])

    questions = Variable(torch.LongTensor(questions), volatile=inference_only).cuda()
    answers = Variable(torch.LongTensor(answers), volatile=inference_only).cuda()

    batch_size = len(batch)
    hidden = init_hidden(model.num_layers, batch_size, model.hidden_size)

    _, encoder_hidden = model.encoder(questions)
    decoder_output, _ = model.decoder(answers, encoder_hidden, hidden)

    loss = 0
    loss_fn = torch.nn.NLLLoss()
    for i in xrange(batch_size):
        loss += loss_fn(decoder_output[i, :answer_lens[i] - 1], answers[i, 1:answer_lens[i]])

    return loss / batch_size


print 'Building model...'
model = Seq2Seq(VOCAB_SIZE, HIDDEN_SIZE).cuda()

optimizer = SGD(model.parameters(), lr=LR)

train_losses = []
val_losses = []

train = Dataset(TRAIN_FILE_NAME)
val = Dataset(VAL_FILE_NAME)

print 'Training...'
start_time = time.time()
for i in xrange(1, ITERS + 1):
    train_batch = [train.get_random_example() for _ in xrange(BATCH_SIZE)]
    val_batch = [val.get_random_example() for _ in xrange(BATCH_SIZE)]

    train_loss = get_loss(model, train_batch)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    val_loss = get_loss(model, val_batch, inference_only=True)

    train_losses.append(train_loss.data[0])
    val_losses.append(val_loss.data[0])

    if i % PRINT_ITERS == 0:
        end_time = time.time()
        string = 'epoch: {}, iters: {}, train loss: {:.2f}, val loss: {:.2f}, time: {:.2f} s'
        print string.format(i / len(train), i, train_loss.data[0], val_loss.data[0], end_time - start_time)
        start_time = time.time()

plot_loss(train_losses, val_losses)