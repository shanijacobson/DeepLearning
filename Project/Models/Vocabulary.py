from collections import Counter, OrderedDict
from torchtext import vocab


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class GlossVocabulary(vocab.Vocab):
    def __init__(self, data):
        vocabulary = build_vocab_from_data(data, key=lambda sample: sample[1])
        self.specials_tokens = []
        super().__init__(vocabulary)


class WordVocabulary(vocab.Vocab):
    def __init__(self, data):
        vocabulary = build_vocab_from_data(data, key=lambda sample: sample[2])
        self.specials_tokens = []
        super().__init__(vocabulary)


def build_vocab_from_data(data, key, min_freq=1) -> vocab.Vocab:
    counter = Counter()
    for sample in data:
        counter.update(key(sample))
    ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    return vocab.vocab(ordered_dict, min_freq=min_freq)

