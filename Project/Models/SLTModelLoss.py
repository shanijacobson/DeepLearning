from torch import nn
import torch
from Models.GlossesTags import TAGS_LIST


class SLTModelLoss(nn.Module):
    def __init__(self, gloss_blank_index, word_ignore_index, gloss_loss_weight=1.0, word_loss_weight=1.0):
        super().__init__()
        self.gloss_loss_weight = gloss_loss_weight
        self.word_loss_weight = word_loss_weight
        self.recognition_loss = nn.CTCLoss(reduction="sum", blank=gloss_blank_index, zero_infinity=True)
        self.translation_loss = nn.NLLLoss(reduction="sum", ignore_index=word_ignore_index)

    def forward(self, glosses, words, glosses_output, words_output, frames_len, glosses_len):
        # Log_probs shape: (input_length, batch_size, gloss_vocab_size)
        recognition_loss = self.recognition_loss(log_probs=glosses_output.permute(1, 0, 2), targets=glosses,
                                                 input_lengths=frames_len, target_lengths=glosses_len)
        # Input shape: (batch_size, word_vocab_size, input_length)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1)[:, :, :-1], target=words[:, 1:])
        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss, recognition_loss, translation_loss


class ModifiedSLTModelLoss(nn.Module):
    def __init__(self, gloss_blank_index, word_ignore_index, glosses_tag, tag_booster_factor=0.3, gloss_loss_weight=1.0, word_loss_weight=1.0):
        super().__init__()
        self.gloss_blank_index = gloss_blank_index
        self.gloss_loss_weight = gloss_loss_weight
        self.word_loss_weight = word_loss_weight
        self.glosses_tag = glosses_tag
        self.tag_booster_factor = tag_booster_factor
        self.recognition_loss = nn.CTCLoss(reduction="sum", blank=gloss_blank_index, zero_infinity=True)
        self.translation_loss = nn.NLLLoss(reduction="sum", ignore_index=word_ignore_index)

    def forward(self, glosses, words, glosses_scores, words_output, frames_len, glosses_len):
        # Log_probs shape: (input_length, batch_size, gloss_vocab_size)
        # tags = [self.glosses_tag[g] for g in glosses]

        modified_glosses_scores = glosses_scores if not self.training else self._get_booster_scores(glosses_scores)
        glosses_probs = modified_glosses_scores.log_softmax(dim=-1)
        recognition_loss = self.recognition_loss(glosses_probs.permute(1, 0, 2), glosses, frames_len, glosses_len)
        #
        # a = torch.all(tmp == recognition_loss)

        # Input shape: (batch_size, word_vocab_size, input_length)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1)[:, :, :-1], target=words[:, 1:])
        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss, recognition_loss, translation_loss

    def _get_booster_scores(self, glosses_scores):
        tags_scores = {}
        for tag_name in TAGS_LIST:
            tmp = torch.tensor([self.glosses_tag[i] == tag_name for i in range(glosses_scores.shape[-1])])
            counter = torch.sum(tmp.type(torch.int))
            tags_scores[tag_name] = torch.sum(torch.where(tmp, glosses_scores, 0), dim=-1) / counter

        return glosses_scores + self.tag_booster_factor * torch.stack(
            [tags_scores[self.glosses_tag[i]] for i in range(glosses_scores.shape[2])], dim=2)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_idx):
    # log_prob (T, N, C) - T: maximum frames size, N: batch size, C: num of glosses
    # target (N, G) - N: batch size, G: maximum glosses size
    # input_lengths + target_lengths (N, ): actual size of frames/ glosses (without padding)
    input_time_size = log_probs.shape[0]
    batch_size = log_probs.shape[1]

    _targets = torch.cat([targets, targets[:, :1]], dim=-1)
    _targets = torch.stack([torch.full_like(_targets, blank_idx), _targets], dim=-1).flatten(start_dim=-2)
    diff_labels = torch.cat([torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
                             _targets[:, 2:] != _targets[:, :-2]], dim=1)

    zero_padding = 2
    zero = torch.tensor(torch.finfo(torch.float32).min, device=log_probs.device, dtype=log_probs.dtype)
    _log_probs = log_probs.gather(-1, _targets.expand(input_time_size, -1, -1).type(torch.int64))
    log_alpha = torch.full((input_time_size, batch_size, zero_padding + _targets.shape[-1]), zero,
                           device=log_probs.device, dtype=log_probs.dtype)
    log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank_idx]
    log_alpha[0, :, zero_padding + 1] = log_probs[0, torch.arange(batch_size), _targets[:, 1].type(torch.long)]

    for t in range(1, input_time_size):
        tmp = torch.stack([log_alpha[t-1, :, 2:], log_alpha[t-1, :, 1: -1],
                           torch.where(diff_labels, log_alpha[t-1, :, :-2], zero)])
        log_alpha[t, :, 2:] = _log_probs[t] + torch.logsumexp(tmp, dim=0)

    l1l2 = log_alpha[input_lengths-1, torch.arange(batch_size)].gather(-1, torch.stack(
        [zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim=-1))
    loss = -torch.logsumexp(l1l2, dim=-1)
    return loss


def try_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank_idx):
    # log_prob (T, N, C) - T: maximum frames size, N: batch size, C: num of glosses
    # target (N, G) - N: batch size, G: maximum glosses size
    # input_lengths + target_lengths (N, ): actual size of frames/ glosses (without padding)
    input_time_size = log_probs.shape[0]
    batch_size = log_probs.shape[1]

    padded_targets = torch.cat([targets, targets[:, :1]], dim=-1)
    padded_targets = torch.stack([torch.full_like(padded_targets, blank_idx), padded_targets], dim=-1).flatten(start_dim=-2).type(torch.long)

    zero_padding = 2
    log_probs_padded_targets = torch.gather(log_probs, dim=-1, index=padded_targets.expand(input_time_size, -1, -1))
    log_alpha = torch.full((input_time_size, batch_size, zero_padding + padded_targets.shape[-1]),
                           torch.finfo(torch.float32).min, device=log_probs.device, dtype=log_probs.dtype)
    log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank_idx]
    log_alpha[0, :, zero_padding + 1] = log_probs[0, torch.arange(batch_size), padded_targets[:, 1]]

    diff_labels = torch.cat([torch.as_tensor([[False, False]], device=targets.device).expand(batch_size, -1),
                             padded_targets[:, 2:] != padded_targets[:, :-2]], dim=1)
    for t in range(1, input_time_size):
        tmp = torch.stack([log_alpha[t-1, :, 2:], log_alpha[t-1, :, 1: -1],
                           torch.where(diff_labels, log_alpha[t-1, :, :-2], torch.finfo(torch.float32).min)])
        log_alpha[t, :, 2:] = log_probs_padded_targets[1, 2][t] + torch.logsumexp(tmp, dim=0)

    l1l2 = torch.gather(log_alpha[input_lengths-1, torch.arange(batch_size)], dim=-1, index=torch.stack(
        [zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim=-1))
    loss = -torch.logsumexp(l1l2, dim=-1)
    return loss
