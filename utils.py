import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from data import EOS_TOKEN
from data import GO_TOKEN
from data import VOCAB_FILE_NAME
from data import load_vocab
from data import seq_to_tokens
from data import tokenize
from data import tokens_to_seq
from models import init_hidden


def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='train')
    plt.plot(val_losses, color='blue', label='val')
    plt.legend(loc='upper right', frameon=False)
    plt.show()


def chat(model):
    vocab = load_vocab(VOCAB_FILE_NAME)
    inverted_vocab = {i: w for w, i in vocab.iteritems()}

    batch_size = 1
    decoder_hidden = init_hidden(model.num_layers, batch_size, model.hidden_size)
    decoder_output = Variable(torch.LongTensor([tokens_to_seq([GO_TOKEN], vocab)]), volatile=True).cuda()

    while True:
        question = raw_input('User: ')
        question = [tokens_to_seq(tokenize(question), vocab)]
        question = Variable(torch.LongTensor(question), volatile=True).cuda()
        _, encoder_hidden = model.encoder(question)

        response = []

        eos = False
        while not eos:
            decoder_output, decoder_hidden = model.decoder(decoder_output, encoder_hidden, decoder_hidden)
            top_i = decoder_output[0, 0].data.topk(1)[1][0]

            decoder_output = Variable(torch.LongTensor([[top_i]]), volatile=True).cuda()
            word = seq_to_tokens([top_i], inverted_vocab)[0]

            if word == EOS_TOKEN:
                eos = True
            else:
                response.append(word)

        print 'Chatbot: {}'.format(' '.join(response))
