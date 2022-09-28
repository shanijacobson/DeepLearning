from torch import nn
import torch
from Models.GlossesTags import TAGS_LIST
from Models import Vocabulary
from Models.GlossesTags import get_glosses_tags


class SLTModelLoss(nn.Module):
    def __init__(self, gloss_vocab, word_ignore_index, gloss_loss_weight=1.0, word_loss_weight=1.0):
        super().__init__()
        self.gloss_blank_index = gloss_vocab[Vocabulary.SIL_TOKEN]
        self.gloss_loss_weight = gloss_loss_weight
        self.word_loss_weight = word_loss_weight
        self.recognition_loss = nn.CTCLoss(reduction="sum", blank=self.gloss_blank_index, zero_infinity=True)
        self.translation_loss = nn.NLLLoss(reduction="sum", ignore_index=word_ignore_index)

    def forward(self, glosses, words, glosses_scores, words_output, frames_len, glosses_len):
        glosses_probs = glosses_scores.log_softmax(dim=-1)
        recognition_loss = self.recognition_loss(glosses_probs.permute(1, 0, 2), glosses, frames_len, glosses_len)

        # Input shape: (batch_size, word_vocab_size, input_length)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1)[:, :, :-1], target=words[:, 1:])
        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss, recognition_loss, translation_loss


class BoosterModelLoss(SLTModelLoss):
    def __init__(self, gloss_vocab, word_ignore_index, gloss_loss_weight=1.0, word_loss_weight=1.0,
                 tag_booster_factor=0.1):
        super().__init__(gloss_vocab, word_ignore_index, gloss_loss_weight, word_loss_weight)
        self.tag_booster_factor = tag_booster_factor
        self.glosses_tag = None if tag_booster_factor == 0 else get_glosses_tags(gloss_vocab)

    def forward(self, glosses, words, glosses_scores, words_output, frames_len, glosses_len):
        # Log_probs shape: (input_length, batch_size, gloss_vocab_size)
        modified_glosses_scores = glosses_scores if not self.training else self._get_booster_scores(glosses_scores)
        glosses_probs = modified_glosses_scores.log_softmax(dim=-1)
        recognition_loss = self.recognition_loss(glosses_probs.permute(1, 0, 2), glosses, frames_len, glosses_len)

        # Input shape: (batch_size, word_vocab_size, input_length)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1)[:, :, :-1], target=words[:, 1:])
        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss, recognition_loss, translation_loss

    def _get_booster_scores(self, glosses_scores):
        if self.tag_booster_factor == 0:
            return glosses_scores
        tags_scores = {}
        for tag_name in TAGS_LIST:
            tmp = torch.tensor([self.glosses_tag[i] == tag_name for i in range(glosses_scores.shape[-1])],
                               device=glosses_scores.device)
            counter = torch.sum(tmp.type(torch.int))
            tags_scores[tag_name] = torch.sum(torch.where(tmp, glosses_scores, 0), dim=-1) / counter

        return glosses_scores + self.tag_booster_factor * torch.stack(
            [tags_scores[self.glosses_tag[i]] for i in range(glosses_scores.shape[2])], dim=2)


class EmotionModelLoss(SLTModelLoss):
    def __init__(self, gloss_vocab, word_ignore_index, gloss_loss_weight=1.0, word_loss_weight=1.0,
                 emo_loss_weight=0, emo_train=0, gloss_train=0, decoder_train=0):
        super().__init__(gloss_vocab, word_ignore_index, gloss_loss_weight, word_loss_weight)
        self.emo_loss_weight = emo_loss_weight
        self.emo_train = emo_train
        self.gloss_train = gloss_train
        self.decoder_train = decoder_train
        self.emo_loss = EmotionLossMasking()

    def forward(self, glosses, words, glosses_probs, words_output, frames_len, glosses_len, emo, emo_outpout):
        # Log_probs shape: (input_length, batch_size, gloss_vocab_size)
        recognition_loss = self.recognition_loss(glosses_probs.permute(1, 0, 2), glosses, frames_len, glosses_len)

        # Input shape: (batch_size, word_vocab_size, input_length)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1)[:, :, :-1], target=words[:, 1:])

        emo_loss = self.emo_loss(emo, emo_outpout, frames_len)
        if self.emo_train > 0:
            self.emo_train -= 1
            return self.emo_loss_weight * emo_loss, 0, 0, emo_loss

        if self.gloss_train > 0:
            self.gloss_train -= 1
            return self.gloss_loss_weight * recognition_loss, recognition_loss, 0, 0

        if self.decoder_train > 0:
            self.decoder_train -= 1
            return self.word_loss_weight * translation_loss, 0, translation_loss, 0

        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss + self.emo_loss_weight * emo_loss, recognition_loss, translation_loss, emo_loss


class EmotionLossMasking(nn.Module):
    def __init__(self, feature_dim=7):
        super().__init__()
        # self.loss_function = nn.SmoothL1Loss(reduction = "sum")
        self.loss_function = nn.KLDivLoss(reduction="sum")
        self.feature_dim = feature_dim

    def forward(self, target, output, length, epsilon=0.00001):
        total_loss = epsilon
        batch_size = target.shape[0]
        counter = 0
        for index in range(batch_size):
            capture_indexs = (
                        (torch.round(target[index, :length[index]], decimals=4) == round(1 / self.feature_dim, 4)).sum(
                            dim=-1) != self.feature_dim)
            if capture_indexs.sum() > 0:
                total_loss += self.loss_function(output[index, :length[index]][capture_indexs].double(),
                                                 target[index, :length[index]][capture_indexs].double())
                counter += 1
        return total_loss / counter
