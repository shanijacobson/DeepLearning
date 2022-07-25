import torch 
import numpy as np
import torch.nn as nn
import math
from Models.Embedding import PositionalEncoding, AlexNetSE
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder



class Encoder(nn.Module):
    def __init__(self, emb_dim, num_layers, nhead, gloss_dim, dim_ff = 2048, drop_p = 0, spatial_flag = False):
        
        super(Encoder,self).__init__()
        self.model_type = 'TransformerEncoder'
        self.pos_encoder = PositionalEncoding(emb_dim, drop_p)
        encoder_layers = TransformerEncoderLayer(emb_dim,nhead,dim_ff,drop_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.spatial_flag = spatial_flag
        self.SE = AlexNetSE(drop_p)
        self.fc = nn.Linear(emb_dim,gloss_dim)
        ### not sure about the dim
        self.sm = nn.Softmax(dim = 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)


    def forward(self, X):
        if self.spatial_flag:
            src = self.SE(X) * math.sqrt(self.d_model)
        else:
            src = X.clone()
        output = self.transformer_encoder(src)
        output_gloss = self.fc(output)
        output_gloss = self.sm(output_gloss)
        return output, output_gloss

class Decoder(nn.Module):
    def __init__(self, emb_dim, num_layers, nhead, trans_dim, dim_ff = 2048, drop_p = 0):
        
        super(Decoder,self).__init__()
        self.model_type = 'TransformerDecoder'
        self.word_encoding = nn.Embedding(trans_dim,emb_dim)
        decoder_layers = TransformerDecoderLayer(emb_dim,nhead,dim_ff,drop_p)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.fc = nn.Linear(emb_dim,trans_dim)
        self.sm = nn.Softmax(dim = 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_encoding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)


    def forward(self,tgt,memory,tgt_mask, memory_mask):
        emb = self.word_encoding(tgt)
        output = self.transformer_decoder(emb, memory, tgt_mask)
        output = self.fc(output)
        output = self.sm(output)
        return output

class TransformerSLT(nn.Module):
    def __init__(self, emb_dim, num_layers_encoder,num_layers_decoder,gloss_dim,trans_dim, nhead, dim_ff = 2048, drop_p = 0,spatial_flag=False):
        super(TransformerSLT,self).__init__()
        self.encoder = Encoder(emb_dim,num_layers_encoder,nhead,gloss_dim,dim_ff,drop_p,spatial_flag)
        self.decoder = Decoder(emb_dim,num_layers_decoder,nhead,trans_dim, dim_ff,drop_p)
    
    def mask_seq(self, seq):
        mask_seq = torch.ones(seq.shape[0],seq.shape[0])
        mask_seq = torch.triu(mask_seq,diagonal=1)
        return mask_seq

    def forward(self, data):
        src, src_gloss = self.encoder(data[0])
        mask_src = self.mask_seq(src)
        mask_target = self.mask_seq(data[2])
        output = self.decoder(data[2], src, mask_target, mask_src)
        return output
    