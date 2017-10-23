from collections import Counter
from itertools import chain

PAD_LABEL = 0
UNK_LABEL = 1
SOS_LABEL = 2
EOS_LABEL = 3
NUM_SPECIAL_LABELS = 4


class Vocab(object):
    def __init__(self, data, max_vocab_size=None):
        if max_vocab_size is not None:
            max_vocab_size -= NUM_SPECIAL_LABELS

        tokens = chain(*chain(*data))
        most_common = [token for token, _ in Counter(tokens).most_common(max_vocab_size)]
        self.token_to_label = {token: i + NUM_SPECIAL_LABELS for i, token in enumerate(most_common)}
        self.label_to_token = {label: token for token, label in self.token_to_label.iteritems()}

    def __len__(self):
        return NUM_SPECIAL_LABELS + len(self.token_to_label)

    def label_encode(self, tokens, use_sos=False):
        labels = [self.token_to_label[t] if t in self.token_to_label else UNK_LABEL for t in tokens]
        labels.append(EOS_LABEL)
        if use_sos:
            labels.insert(0, SOS_LABEL)
        return labels

    def label_decode(self, labels):
        tokens = []
        for l in labels:
            if l in self.label_to_token:
                tokens.append(self.label_to_token[l])
            elif l == UNK_LABEL:
                tokens.append('<unk>')
        return tokens
