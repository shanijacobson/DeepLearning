from Models.SignGlossLanguage import SignGlossLanguage
from Models.loss import SLtLoss
from Models.Vocabulary import GlossVocabulary, WordVocabulary
from torch.utils.data import DataLoader, random_split
import torch
from Models.TransformerModel import Encoder, TransformerSLT, DEVICE
import os

DATA_PATH = os.path.join("Data", "Phoenix14")
BATCH_SIZE = 64
VALIDATION_SIZE = 520


def train_batch(batch):
    pass


def initialize():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(os.path.join("Data", "models")):
        os.makedirs(os.path.join("Data", "models"))

def train_model():
    initialize()

    # Build Vocabularies:
    gloss_vocab = GlossVocabulary(DATA_PATH)
    word_vocab = WordVocabulary(DATA_PATH)

    # Build Datasets
    total_train_dataset = SignGlossLanguage(root=DATA_PATH, train=True, download=False,
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
    # TODO: we should start work with google colab, its impossibale to train this modle on the leptop
  #  model = Encoder(1024, 3, 4, gloss_ dim=len(gloss_vocab)).to(DEVICE)
    model = TransformerSLT(1024, 2, 2, len(gloss_vocab), len(word_vocab), 4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = SLtLoss(0.3,0.7,gloss_vocab.get_itos().index("<pad>"))
    for _ in range(3):
        lost_list = []
        for frame, gloss, words in train_loader:
            # TODO: finding the real length of every input seq (without pading) , that what i understood as the input_length
            input_size = torch.tensor(list(map(lambda x: (x.sum(axis=1)!=0).sum(),frame)))
            frame = frame.permute(1, 0, 2).to(DEVICE) # TODO: why can't batch size be in the first coordinate? Can you change in the model? If no lets move this to the dataset model
            gloss = gloss.T.to(DEVICE)
            words = words.T.to(DEVICE)
            #  gloss = torch.tensor([[[1 if token == elm else 0 for elm in gloss_vocab.get_itos()] for token in sent] for sent in gloss])
            # gloss = torch.tensor([[gloss_vocab.get_itos().index(token) for token in sent] for sent in gloss])
            # words = torch.tensor([[word_vocab.get_itos().index(token) for token in sent] for sent in words])

            # words = torch.tensor([[[1 if token == elm else 0 for elm in word_vocab.get_itos()] for token in sent] for sent in words])
            # gloss = gloss.permute(1,0,2)
            # words = words.permute(1,0,2)
            words_pred, gloss_pred = model((frame, gloss, words))
            loss = criterion(gloss, words, gloss_pred, words_pred,input_size)
            lost_list.append(float(loss) / frame.shape[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(sum(lost_list))
        

if __name__ == '__main__':
    train_loader, gloss_vocab, word_vocab = train_model()
    test(train_loader, gloss_vocab, word_vocab)
