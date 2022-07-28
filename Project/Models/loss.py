import torch 
import torch.nn as nn
from Models.TransformerModel import DEVICE

class SLtLoss(nn.Module):
    #TODO: you should take another look at this class, i did make some asumption about the input, that im not sure about 
    def __init__(self, lambda_g, lambda_W, blank_token) :
        super().__init__()
        self.lambda_g = lambda_g
        self.lambda_w = lambda_W
        self.blank_token = blank_token
        self.recognition_loss = nn.CTCLoss(blank=blank_token, zero_infinity=True).to(DEVICE)
        # TODO : input the blanks indexs to the loss function 
        self.translation_loss = nn.NLLLoss( reduction="sum").to(DEVICE)
    
    def forward(self,gloss, words, output_gloss, output_word, input_lengths):
        # TODO : claculating the unpad length of each seq in the target, that what i understode as the target_length 
        tgt_gloss_length = torch.tensor(list(map(lambda x: sum(x!=self.blank_token), gloss.T)))
        loss = self.lambda_g * self.recognition_loss(output_gloss, gloss.T,input_lengths, tgt_gloss_length)
        # TODO : I understend that we need to flatten the seq dim , we should double chack me if you have the time
        seq_size, batch_size, feature_size = output_word.shape
        output_word = output_word.permute(1,0,2).reshape(batch_size * seq_size, feature_size).to(DEVICE)
        words = words.T.flatten().type(torch.LongTensor).to(DEVICE)
        loss += self.lambda_w * self.translation_loss(output_word,words)
        return loss
    
    # TODO: build label smooth function 

    