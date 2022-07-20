import Models.SignGlossLanguage as sgl
import Models.Vocabulary as vocab
import os
from torch.utils.data import DataLoader

DATA_PATH = os.path.join("Data", "Phoenix14")


def train_batch(batch):
    pass


def train_model():
    train_dataset = sgl.SignGlossLanguage(root=DATA_PATH, train=True, download=True)
    test_dataset = sgl.SignGlossLanguage(root=DATA_PATH, train=False, max_words=train_dataset.max_words,
                                         max_glosses=train_dataset.max_glosses,
                                         max_signs_frames=train_dataset.max_signs_frames)
    # gloss_vocab = vocab.GlossVocabulary(data=test_dataset)
    # word_vocab = vocab.WordVocabulary(data=test_dataset)
    # test_loader = DataLoader(test_dataset, 4, shuffle=True)
    # for _, batch in enumerate(test_loader):
    #     train_batch(batch)


if __name__ == '__main__':
    train_model()
