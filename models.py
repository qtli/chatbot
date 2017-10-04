import torch
import torch.nn as nn
from torch.autograd import Variable


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, num_layers):
        super(Seq2Seq, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = EncoderRNN(input_vocab_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(output_vocab_size, hidden_size, num_layers)


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, self.num_layers, batch_first=True)

    def forward(self, input_seqs):
        input_seqs = self.embedding(input_seqs)
        batch_size = input_seqs.size()[0]
        output, hidden = self.gru(input_seqs, init_hidden(self.num_layers, batch_size, self.hidden_size))
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, self.num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()

    @staticmethod
    def create_rnn_input(embedded, thought):
        # reorder axes to be (seq_len, batch_size, hidden_size)
        embedded = embedded.permute(1, 0, 2)

        seq_len, batch_size, hidden_size = embedded.size()
        rnn_input = Variable(torch.zeros((seq_len, batch_size, 2 * hidden_size))).cuda()
        for i in xrange(seq_len):
            for j in xrange(batch_size):
                rnn_input[i, j] = torch.cat((embedded[i, j], thought[0, j]))

        # make batch first
        return rnn_input.permute(1, 0, 2)

    def softmax_batch(self, linear_output):
        result = Variable(torch.zeros(linear_output.size())).cuda()
        batch_size = linear_output.size()[0]
        for i in xrange(batch_size):
            result[i] = self.softmax(linear_output[i])
        return result

    def forward(self, target_seqs, thought, hidden):
        target_seqs = self.embedding(target_seqs)
        rnn_input = self.create_rnn_input(target_seqs, thought)
        output, hidden = self.gru(rnn_input, hidden)
        output = self.softmax_batch(self.out(output))
        return output, hidden


def init_hidden(num_layers, batch_size, hidden_size):
    return Variable(torch.zeros(num_layers, batch_size, hidden_size)).cuda()
