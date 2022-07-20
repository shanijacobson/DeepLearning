from Models.SignGlossLanguage import SignGlossLanguage
from Models.Vocabulary import GlossVocabulary, WordVocabulary
from torch.utils.data import DataLoader, random_split
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


if __name__ == '__main__':
    train_model()
