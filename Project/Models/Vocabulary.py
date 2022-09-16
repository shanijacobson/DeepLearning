from collections import Counter, OrderedDict

import torch
from torchtext import vocab
import gzip
import os
import pickle

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SIL_TOKEN = "<sil>"


class GlossVocabulary(vocab.Vocab):
    def __init__(self, root):
        self.root = root
        self.specials_tokens = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        vocabulary = build_vocab_from_data(root, "gloss", specials=self.specials_tokens)
        super().__init__(vocabulary)

    def idx_to_seq(self, idx_list):
        idx_to_glosses = self.get_itos()
        return [idx_to_glosses[idx] for idx in idx_list]


class WordVocabulary(vocab.Vocab):
    def __init__(self, root):
        self.root = root
        self.specials_tokens = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        vocabulary = build_vocab_from_data(root, "text", specials=self.specials_tokens)
        super().__init__(vocabulary)

    def idx_to_seq(self, idx_list, eos_stop=True):
        idx_to_words = self.get_itos()
        seq = []
        for idx in idx_list:
            seq.append(idx_to_words[idx])
            if idx == self.vocab[EOS_TOKEN] and eos_stop:
                break
        return idx_to_words


def build_vocab_from_data(root, key, specials, min_freq=1) -> vocab.Vocab:
    path = os.path.join("Data", "models", f"{key}_vocabulary")
    if os.path.exists(path):
        print(f"Getting existing vocabulary: {key}")
        return torch.load(path)

    counter = Counter()
    source_list = ["phoenix14t.pami0.train"]

    for name in source_list:
        with gzip.open(os.path.join(root, name), "rb") as f:
            loaded_object = pickle.load(f)
            for sample in loaded_object:
                if sample['sign'].shape[0] > 400:
                    continue
                tmp = sample[key].replace('.', '').strip().split(' ')
                if len(tmp) > 400:
                    continue
                tmp = [g for t in tmp for g in t.split("+")]
                counter.update(tmp)
    ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    vocabulary = vocab.vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=True)
    vocabulary.set_default_index(vocabulary.get_stoi()[UNK_TOKEN])
    torch.save(vocabulary, path)
    return vocabulary
