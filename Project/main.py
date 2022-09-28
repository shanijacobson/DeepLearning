from Models.Predictions import beam_search, greedy
from Models.GlossesTags import get_glosses_tags
from Models.SignGlossLanguage import SignGlossLanguage
from Models.SLTModelLoss import SLTModelLoss
from Models.Vocabulary import GlossVocabulary, WordVocabulary, PAD_TOKEN, SIL_TOKEN, BOS_TOKEN, EOS_TOKEN
from torch.utils.data import DataLoader, random_split
import torch
from Models.TransformerModel import SLTModel
import os

DATA_PATH = os.path.join("Data", "Phoenix14")
BATCH_SIZE = 32
VALIDATION_SIZE = 520
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_batch(batch):
    pass


def initialize():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(os.path.join("Data", "models")):
        os.makedirs(os.path.join("Data", "models"))


def get_data():
    initialize()

    # Build Vocabularies:
    gloss_vocab = GlossVocabulary(DATA_PATH)
    word_vocab = WordVocabulary(DATA_PATH)

    # Build Datasets
    train_dataset = SignGlossLanguage(root=DATA_PATH, type="train", download=False,
                                      word_vocab=word_vocab.vocab, gloss_vocab=gloss_vocab.vocab)
    valid_dataset = SignGlossLanguage(root=DATA_PATH, type="dev", download=False,
                                      word_vocab=word_vocab.vocab, gloss_vocab=gloss_vocab.vocab)
    test_dataset = SignGlossLanguage(root=DATA_PATH, type="test", download=False,
                                     word_vocab=word_vocab, gloss_vocab=gloss_vocab)

    # Data Loaders:
    train_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=True)
    # validation_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
    return train_loader, gloss_vocab, word_vocab


def train_model():
    valid_loader, gloss_vocab, word_vocab = get_data()

    model = SLTModel(frame_size=1024, gloss_dim=len(gloss_vocab), words_dim=len(word_vocab),
                     word_padding_idx=word_vocab[PAD_TOKEN]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    gloss_tag = get_glosses_tags(gloss_vocab)
    criterion = SLTModelLoss(gloss_vocab, word_vocab[PAD_TOKEN], tag_booster_factor=0.3).to(DEVICE)
    idx_to_words = word_vocab.get_itos()
    iter = 0
    for _ in range(200):
        lost_list = []
        txt_hyp = []

        for (frames, frames_len), (glosses, glosses_len), (words, words_len) in valid_loader:
            print(iter)
            iter += 1
            frames = frames.to(DEVICE)
            glosses = glosses.to(DEVICE)
            words = words.to(DEVICE)
            words_output, glosses_probs, glosses_scores, encoder_output = model(frames, words)
            predict = greedy(model, frames, words, encoder_output, word_vocab[BOS_TOKEN],
                             word_vocab[EOS_TOKEN], word_vocab[PAD_TOKEN], max_output_length=30)

            idx_to_seq = [' '.join([idx_to_words[idx] for idx in seq]) for seq in predict]
            txt_hyp.extend(idx_to_seq)
            # loss = criterion(glosses, words.T, glosses_output, words_output, frames_len, glosses_len)
            # lost_list.append(float(loss) / frames.shape[1])
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            if ((iter + 1) % 30) == 0:
                # predict = greedy(1, frames.shape[0], model, frames, encoder_output, 30, word_vocab[BOS_TOKEN],
                #                       word_vocab[PAD_TOKEN])

                predict_2 = greedy(model, frames, words, encoder_output, word_vocab[BOS_TOKEN], word_vocab[EOS_TOKEN],
                                   word_vocab[PAD_TOKEN])

            loss = criterion(glosses, words, glosses_scores, words_output, frames_len, glosses_len)
            total_loss = torch.sum(loss[0]) + torch.sum(loss[1])
            norm_loss = total_loss / glosses.shape[0]
            optimizer.zero_grad()
            norm_loss.backward()
            # print(loss.cpu().detach().numpy())
            optimizer.step()
        print(sum(lost_list))


if __name__ == '__main__':
    train_model()
