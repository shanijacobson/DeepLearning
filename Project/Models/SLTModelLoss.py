from turtle import forward
from torch import nn
import torch

class SLTModelLoss(nn.Module):
    def __init__(self, gloss_blank_index, word_ignore_index, gloss_loss_weight=1.0, word_loss_weight=1.0, emo_loss_weight = 0, emo_train = 0, gloss_train = 0):
        super().__init__()
        self.gloss_loss_weight = gloss_loss_weight
        self.word_loss_weight = word_loss_weight
        self.emo_loss_weight = emo_loss_weight
        self.emo_train = emo_train
        self.gloss_train = gloss_train
        self.recognition_loss = nn.CTCLoss(reduction="sum", blank=gloss_blank_index, zero_infinity=True)
        self.translation_loss = nn.NLLLoss(reduction="sum", ignore_index=word_ignore_index)
        self.emo_loss = EmoationLossMasking()

    def forward(self, glosses, words, glosses_output, words_output, frames_len, glosses_len, emo =None, emo_outpout = None):
        # Log_probs shape: (input_length, batch_size, gloss_vocab_size)
        recognition_loss = self.recognition_loss(log_probs=glosses_output.permute(1, 0, 2), targets=glosses,
                                                 input_lengths=frames_len, target_lengths=glosses_len)
        # Input shape: (batch_size, word_vocab_size, input_length)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1), target=words)
        translation_loss = self.translation_loss(input=words_output.permute(0, 2, 1)[:, :, :-1], target=words[:, 1:])
        # return total loss, loss recognition and loss translation (for logging)
        if self.gloss_train > 0:
            self.gloss_train -= 1
            return  self.gloss_loss_weight * recognition_loss, recognition_loss, translation_loss, None

        if emo is not None:
            emo_loss = self.emo_loss(emo,emo_outpout,frames_len)
            if self.emo_train > 0:
                 self.emo_train -= 1
                 return  self.emo_loss_weight * emo_loss, recognition_loss, translation_loss ,emo_loss

            return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss  + self.emo_loss_weight * emo_loss, recognition_loss, translation_loss ,emo_loss
        return self.gloss_loss_weight * recognition_loss + self.word_loss_weight * translation_loss, recognition_loss, translation_loss, None



class EmoationLossMasking(nn.Module):
    def __init__(self, feature_dim = 7):
        super().__init__()
        #self.loss_function = nn.SmoothL1Loss(reduction = "sum")
        self.loss_function = nn.KLDivLoss(reduction="sum")
        self.feature_dim = feature_dim
    
    def forward(self,target, output, length, epsilon = 0.00001):
        total_loss = epsilon
        batch_size = target.shape[0]
        counter = 0
        for index in range(batch_size):
            capture_indexs = ((torch.round(target[index,:length[index]],decimals=4) == round(1/self.feature_dim,4)).sum(dim=-1) != self.feature_dim)
            if capture_indexs.sum() > 0:
               total_loss += self.loss_function(output[index,:length[index]][capture_indexs].double(), target[index,:length[index]][capture_indexs].double())
               counter += 1
        return total_loss / counter