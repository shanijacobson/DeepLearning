from collections import Counter, OrderedDict

import torch
from torchtext import vocab
import gzip
import os
import pickle
import numpy as np

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SIL_TOKEN = "<sil>"


class GlossVocabulary(vocab.Vocab):
    def __init__(self, root, based_tags=False):
        self.root = root
        self.specials_tokens = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
        vocabulary = build_gloss_vocab_from_tagging(root, specials=self.specials_tokens) if based_tags \
            else build_vocab_from_data(root, "gloss", specials=self.specials_tokens)
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
                tmp = sample[key].strip().split(' ')
                if len(tmp) > 400:
                    continue
                counter.update(tmp)

    ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    vocabulary = vocab.vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=True)
    vocabulary.set_default_index(vocabulary.get_stoi()[UNK_TOKEN])
    torch.save(vocabulary, path)
    return vocabulary


def build_gloss_vocab_from_tagging(root, specials, min_freq=1) -> vocab.Vocab:
    path = os.path.join("Data", "models", "gloss_based_tags_vocabulary")
    if os.path.exists(path):
        print(f"Getting existing vocabulary: glosses from tags")
        return torch.load(path)
    tags_path = f"Data/models/glosses_tags.npy"
    if not os.path.exists(tags_path):
        print("Cannot build gloss vocab from tags: tags files does not exist. Build vocab from data.")
        return build_vocab_from_data(root, "gloss", specials, min_freq=1)
    tags = np.load(tags_path, allow_pickle=True).item()
    counter = Counter()
    source_list = ["phoenix14t.pami0.train"]

    for name in source_list:
        with gzip.open(os.path.join(root, name), "rb") as f:
            loaded_object = pickle.load(f)
            for sample in loaded_object:
                if sample['sign'].shape[0] > 400:
                    continue

                glosses = []
                for g in sample["gloss"].strip().split(' '):
                    if g in tags.keys() or '-' not in g:
                        glosses.append(g)
                    else:
                        if any(gloss_split in tags.keys() for gloss_split in g.split('-')):
                            glosses.extend(g.split('-'))
                        else:
                            glosses.append(g)
                if len(glosses) > 400:
                    continue

                counter.update(glosses)

    ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
    vocabulary = vocab.vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=True)
    vocabulary.set_default_index(vocabulary.get_stoi()[UNK_TOKEN])
    torch.save(vocabulary, path)
    return vocabulary
