from collections import Counter, OrderedDict

import torch
from torchtext import vocab
import gzip
import os
import pickle


PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SIL_TOKEN = "<sil>"


class GlossVocabulary(vocab.Vocab):
    def __init__(self, root):
        self.root = root
        vocabulary = build_vocab_from_data(root, "gloss")
        self.specials_tokens = [SIL_TOKEN, PAD_TOKEN, EOS_TOKEN]
        [vocabulary.append_token(token) for token in self.specials_tokens]
        super().__init__(vocabulary)


class WordVocabulary(vocab.Vocab):
    def __init__(self, root):
        self.root = root
        vocabulary = build_vocab_from_data(root, "text")
        self.specials_tokens = [PAD_TOKEN, EOS_TOKEN]
        [vocabulary.append_token(token) for token in self.specials_tokens]
        super().__init__(vocabulary)


def build_vocab_from_data(root, key, min_freq=1) -> vocab.Vocab:
    path = os.path.join("Data", "models", f"{key}_vocabulary")
    if os.path.exists(path):
        print(f"Getting existing vocabulary: {key}")
        return torch.load(path)

    counter = Counter()
    source_list = ["phoenix14t.pami0.train", "phoenix14t.pami0.dev", "phoenix14t.pami0.test"]

    for name in source_list:
        with gzip.open(os.path.join(root, name), "rb") as f:
            loaded_object = pickle.load(f)
            for sample in loaded_object:
                counter.update(sample[key].split(' '))
    ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    vocabulary = vocab.vocab(ordered_dict, min_freq=min_freq)
    torch.save(vocabulary, path)
    return vocabulary
