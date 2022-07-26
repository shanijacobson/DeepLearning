from Models.SignGlossLanguage import SignGlossLanguage
from Models.Vocabulary import GlossVocabulary, WordVocabulary
from torch.utils.data import DataLoader, random_split
import torch
from Models.TransformerModel import Encoder, TransformerSLT
import os

DATA_PATH = os.path.join("Data", "Phoenix14")
BATCH_SIZE = 64
VALIDATION_SIZE = 520


def train_batch(batch):
    pass


def train_model():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    total_train_dataset = SignGlossLanguage(root=DATA_PATH, train=True, download=True)
    train_dataset, valid_dataset = random_split(total_train_dataset,
                                                [len(total_train_dataset) - VALIDATION_SIZE, VALIDATION_SIZE])
    test_dataset = SignGlossLanguage(root=DATA_PATH, train=False, download=True)

    # Data Loaders:
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # Vocabularies:
    gloss_vocab = GlossVocabulary(data=total_train_dataset)
    word_vocab = WordVocabulary(data=total_train_dataset)
    return train_loader, gloss_vocab, word_vocab


def test(train_loader, gloss_vocab, word_vocab):
    model = Encoder(1024, 3, 4, gloss_dim=len(gloss_vocab))
    model = TransformerSLT(1024, 3, 3, len(gloss_vocab), len(word_vocab), 4)
    for batch in train_loader:
        frame = batch[0].permute(1, 0, 2)
        gloss = batch[1]
        words = batch[2]
        #  gloss = torch.tensor([[[1 if token == elm else 0 for elm in gloss_vocab.get_itos()] for token in sent] for sent in gloss])
        gloss = torch.tensor([[gloss_vocab.get_itos().index(token) for token in sent] for sent in gloss])
        words = torch.tensor([[word_vocab.get_itos().index(token) for token in sent] for sent in words])

        # words = torch.tensor([[[1 if token == elm else 0 for elm in word_vocab.get_itos()] for token in sent] for sent in words])
        # gloss = gloss.permute(1,0,2)
        # words = words.permute(1,0,2)
        model((frame, gloss, words))


if __name__ == '__main__':
    train_loader, gloss_vocab, word_vocab = train_model()
    test(train_loader, gloss_vocab, word_vocab)
