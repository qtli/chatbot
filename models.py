import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD

from data import TRAIN_FILE_NAME
from data import VAL_FILE_NAME
from dataset import Dataset


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = DecoderRNN(vocab_size, hidden_size)

    def _get_loss(self, example):
        loss_fn = torch.nn.NLLLoss()

        input_seqs = Variable(torch.LongTensor([example['question']]))
        target_seqs = Variable(torch.LongTensor([example['answer']]))
        output = self(input_seqs, target_seqs)
        return loss_fn(output[0][:-1], target_seqs[0][1:])

    def forward(self, input_seqs, target_seqs):
        _, encoder_hidden = self.encoder(input_seqs)
        decoder_output, _ = self.decoder(target_seqs, encoder_hidden)
        return decoder_output

    def train(self, lr=1e-3, iters=7500, print_iters=100):
        optimizer = SGD(self.parameters(), lr=lr)

        train_losses = []
        val_losses = []

        train = Dataset(TRAIN_FILE_NAME)
        val = Dataset(VAL_FILE_NAME)

        start_time = time.time()
        for i in xrange(1, iters + 1):
            train_loss = self._get_loss(train.get_random_example())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            val_loss = self._get_loss(val.get_random_example())

            train_losses.append(train_loss.data[0])
            val_losses.append(val_loss.data[0])

            if i % print_iters == 0:
                end_time = time.time()
                string = 'epoch: {}, iters: {}, train loss: {:.2f}, val loss: {:.2f}, time: {:.2f} s'
                print string.format(i / len(train), i, train_loss.data[0], val_loss.data[0], end_time - start_time)
                start_time = time.time()

        return train_losses, val_losses


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.num_layers = 1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, batch_first=True)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, input_seqs):
        input_seqs = self.embedding(input_seqs)
        batch_size = input_seqs.size()[0]
        output, hidden = self.gru(input_seqs, self.init_hidden(batch_size))
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(DecoderRNN, self).__init__()

        self.num_layers = 1
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    @staticmethod
    def create_rnn_input(embedded, thought):
        # reorder axes to be (seq_len, batch, hidden)
        embedded = embedded.permute(1, 0, 2)
        # thought = thought.unsqueeze(0)

        seq_len, batch, hidden = embedded.size()
        rnn_input = Variable(torch.zeros((seq_len, batch, 2 * hidden)))
        for i in xrange(seq_len):
            for j in xrange(batch):
                rnn_input[i, j] = torch.cat((embedded[i, j], thought[0, j]))

        # make batch first
        return rnn_input.permute(1, 0, 2)

    def softmax_over_time(self, linear_output):
        result = Variable(torch.zeros(linear_output.size()))
        batch, seq_len, vocab = linear_output.size()
        for i in xrange(batch):
            result[i] = self.softmax(linear_output[i])
        return result

    def forward(self, target_seqs, thought):
        target_seqs = self.embedding(target_seqs)
        rnn_input = self.create_rnn_input(target_seqs, thought)
        batch_size = target_seqs.size()[0]
        output, hidden = self.gru(rnn_input, self.init_hidden(batch_size))
        output = self.softmax_over_time(self.out(output))
        return output, hidden
