from Models.SignGlossLanguage import SignGlossLanguage
from Models.Vocabulary import GlossVocabulary, WordVocabulary
from torch.utils.data import DataLoader, random_split
from Models.TransformerModel import Encoder, TransformerSLT
import os

DATA_PATH = os.path.join("Data", "Phoenix14")
BATCH_SIZE = 10
VALIDATION_SIZE = 519


def initialize():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(os.path.join(DATA_PATH, "Video")):
        os.makedirs(os.path.join(DATA_PATH, "Video"))
    if not os.path.exists(os.path.join("Data", "models")):
        os.makedirs(os.path.join("Data", "models"))


def train_model():
    initialize()

    # Build Vocabularies:
    gloss_vocab = GlossVocabulary(DATA_PATH)
    word_vocab = WordVocabulary(DATA_PATH)

    # Build Datasets
    total_train_dataset = SignGlossLanguage(root=DATA_PATH, train=True, download=False, original_frames=False,
                                            word_vocab=word_vocab, gloss_vocab=gloss_vocab)
    train_dataset, valid_dataset = random_split(total_train_dataset,
                                                [len(total_train_dataset) - VALIDATION_SIZE, VALIDATION_SIZE])
    test_dataset = SignGlossLanguage(root=DATA_PATH, train=False, download=False,
                                     word_vocab=word_vocab, gloss_vocab=gloss_vocab)

    # Data Loaders:
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
    return train_loader, gloss_vocab, word_vocab


def test(train_loader, gloss_vocab, word_vocab):
    model = Encoder(1024, 3, 4, gloss_dim=len(gloss_vocab))
    model = TransformerSLT(1024, 3, 3, len(gloss_vocab), len(word_vocab), 4, spatial_flag=False)
    for frame, gloss, words in train_loader:
        # TODO: Check if batch size can be in the first coordinate. If no move this to the dataset model
        frame = frame.permute(1, 0, 2, 3)
        gloss = gloss.T
        words = words.T
        model(frame, gloss, words)


if __name__ == '__main__':
    train_loader, gloss_vocab, word_vocab = train_model()
    test(train_loader, gloss_vocab, word_vocab)
