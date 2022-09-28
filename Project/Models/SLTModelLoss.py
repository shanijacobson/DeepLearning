from torch import nn
import torch
from Models.GlossesTags import TAGS_LIST
from Models import Vocabulary
from Models.GlossesTags import get_glosses_tags


class SLTModelLoss(nn.Module):
    def __init__(self, gloss_vocab, word_ignore_index, gloss_loss_weight=1.0, word_loss_weight=1.0,
                 tag_booster_factor=0):
        super().__init__()
        self.gloss_blank_index = gloss_vocab[Vocabulary.SIL_TOKEN]
        self.gloss_loss_weight = gloss_loss_weight
        self.word_loss_weight = word_loss_weight
        self.recognition_loss = nn.CTCLoss(reduction="sum", blank=self.gloss_blank_index, zero_infinity=True)
        self.translation_loss = nn.NLLLoss(reduction="sum", ignore_index=word_ignore_index)
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
