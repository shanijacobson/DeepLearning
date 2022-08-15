from torch import nn


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
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1), target=words)
        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss
