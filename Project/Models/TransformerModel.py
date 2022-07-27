import torch
import torch.nn as nn
import math
from Models.Embedding import AlexNetSE
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class Encoder(nn.Module):
    def __init__(self, emb_dim, num_layers, nhead, gloss_dim, dim_ff=2048, drop_p=0):

        super(Encoder, self).__init__()
        self.model_type = 'TransformerEncoder'
        encoder_layers = TransformerEncoderLayer(emb_dim, nhead, dim_ff, drop_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(emb_dim, gloss_dim)
        self.sm = nn.Softmax(dim=2)  # TODO: not sure about the dim
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, X):
        output = self.transformer_encoder(X)
        output_gloss = self.fc(output)
        output_gloss = self.sm(output_gloss)
        return output, output_gloss


class Decoder(nn.Module):
    def __init__(self, emb_dim, num_layers, nhead, trans_dim, dim_ff=2048, drop_p=0):
        super(Decoder, self).__init__()
        self.model_type = 'TransformerDecoder'
        self.word_encoding = nn.Embedding(trans_dim, emb_dim)
        decoder_layers = TransformerDecoderLayer(emb_dim, nhead, dim_ff, drop_p)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.fc = nn.Linear(emb_dim, trans_dim)
        self.sm = nn.Softmax(dim=2)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.word_encoding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        emb = self.word_encoding(tgt)
        output = self.transformer_decoder(emb, memory, tgt_mask)
        output = self.fc(output)
        output = self.sm(output)
        return output


class TransformerSLT(nn.Module):
    def __init__(self, emb_dim, num_layers_encoder, num_layers_decoder, gloss_dim, trans_dim, nhead, dim_ff=2048,
                 drop_p=0, spatial_flag=False):
        super(TransformerSLT, self).__init__()
        self.spatial_embedding = AlexNetSE(drop_p) if spatial_flag else None
        self.encoder = Encoder(emb_dim, num_layers_encoder, nhead, gloss_dim, dim_ff, drop_p)
        self.decoder = Decoder(emb_dim, num_layers_decoder, nhead, trans_dim, dim_ff, drop_p)

    def mask_seq(self, seq):
        mask_seq = torch.ones(seq.shape[0], seq.shape[0])
        mask_seq = torch.triu(mask_seq, diagonal=1)
        return mask_seq

    def forward(self, frames, glosses, signs):
        if self.spatial_embedding:
            src = self.spatial_embedding(frames) * math.sqrt(self.d_model)
        else:
            src = frames.squeeze(dim=2).clone()
        src, src_gloss = self.encoder(src)
        mask_src = self.mask_seq(src)
        mask_target = self.mask_seq(signs)
        output = self.decoder(signs, src, mask_target, mask_src)
        return output
