class Vocab(object):
    PAD_LABEL = 0
    UNK_LABEL = 1
    SOS_LABEL = 2
    EOS_LABEL = 3

    PAD_TOKEN = 'pad>'
    UNK_TOKEN = '<unk>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'

    def __init__(self):
        self.label_to_token = {
            Vocab.PAD_LABEL: Vocab.PAD_TOKEN,
            Vocab.UNK_LABEL: Vocab.UNK_TOKEN,
            Vocab.SOS_LABEL: Vocab.SOS_TOKEN,
            Vocab.EOS_LABEL: Vocab.EOS_TOKEN,
        }
        self.token_to_label = {token: label for label, token in self.label_to_token.iteritems()}
        self.vocab_size = len(self.label_to_token)

    def __len__(self):
        return self.vocab_size

    def add_token(self, token):
        if token not in self.token_to_label:
            label = len(self)
            self.token_to_label[token] = label
            self.label_to_token[label] = token
            self.vocab_size += 1

    def label_encode(self, tokens):
        labels = []
        for t in tokens:
            if t in self.token_to_label:
                l = self.token_to_label[t]
            else:
                l = Vocab.UNK_LABEL
            labels.append(l)
        return labels

    def label_decode(self, labels):
        return [self.label_to_token[l] for l in labels]
