import random
from itertools import chain

from vocab import PAD_LABEL


# TODO: update chat function
# def chat(model):
#     vocab = load_vocab(VOCAB_FILE_NAME)
#     inverted_vocab = {i: w for w, i in vocab.iteritems()}
#
#     batch_size = 1
#     decoder_hidden = init_hidden(model.num_layers, batch_size, model.hidden_size)
#     decoder_output = Variable(torch.LongTensor([tokens_to_seq([GO_TOKEN], vocab)]), volatile=True).cuda()
#
#     while True:
#         question = raw_input('User: ')
#         question = [tokens_to_seq(tokenize(question), vocab)]
#         question = Variable(torch.LongTensor(question), volatile=True).cuda()
#         _, encoder_hidden = model.encoder(question)
#
#         response = []
#
#         eos = False
#         while not eos:
#             decoder_output, decoder_hidden = model.decoder(decoder_output, encoder_hidden, decoder_hidden)
#             top_i = decoder_output[0, 0].data.topk(1)[1][0]
#
#             decoder_output = Variable(torch.LongTensor([[top_i]]), volatile=True).cuda()
#             word = seq_to_tokens([top_i], inverted_vocab)[0]
#
#             if word == EOS_TOKEN:
#                 eos = True
#             else:
#                 response.append(word)
#
#         print 'Chatbot: {}'.format(' '.join(response))


def pad_seqs(seqs):
    max_seq_len = max(len(s) for s in seqs)
    result = []
    for s in seqs:
        padding = [PAD_LABEL] * (max_seq_len - len(s))
        result.append(s + padding)
    return result


def split_data(data, train_percent=.6, val_percent=.2):
    random.shuffle(data)

    num_examples = len(data)
    num_train = int(train_percent * num_examples)
    num_val = int(val_percent * num_examples)

    return data[:num_train], data[num_train:num_train + num_val], data[num_train + num_val:]


def save_data(data, encoding, file_path):
    data = [' '.join(line).encode(encoding) for line in chain(*data)]
    with open(file_path, 'w') as f:
        f.write('\n'.join(data))


def load_data(encoding, file_path):
    with open(file_path, 'r') as f:
        file_data = [line.decode(encoding).split() for line in f]

    output_data = []
    for i in xrange(len(file_data) - 1):
        qa_pair = (file_data[i], file_data[i + 1])
        output_data.append(qa_pair)
    return output_data
