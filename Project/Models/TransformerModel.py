import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, n_head, ff_size, drop_p):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb_dropout = nn.Dropout(drop_p)
        self.pos_encoder = PositionalEncoder(embedding_dim, drop_p)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=ff_size,
                                                 dropout=drop_p, batch_first=True, norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers,
                                                      norm=nn.LayerNorm(embedding_dim, eps=1e-6))

    def forward(self, embedded_frames, frames_padding_mask):
        output = self.pos_encoder(embedded_frames)
        output = self.emb_dropout(output)
        return self.transformer_encoder(output, src_key_padding_mask=frames_padding_mask)


class Merger(nn.Module):
    def __init__(self, dim,drop_p, num_layers=1):
        super(Merger, self).__init__()
        self.num_layers = num_layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=2 * dim, out_features=dim),
            nn.ReLU(),
            nn.Dropout(drop_p)
        )
        self.layers_list =  nn.ModuleList([nn.Sequential(
            nn.Linear(in_features= dim, out_features=dim),
            nn.ReLU(),
            nn.Dropout(drop_p)
        ) for _ in range(num_layers-1)])

    def forward(self, embedded_frames, emo_frames):
        merge_input = torch.concat([embedded_frames,emo_frames],dim=-1).to(embedded_frames.device)
        output = self.layer1(merge_input)
        if self.num_layers > 1:
           for layer in self.layers_list:
            output = layer(output)
        return output


class Decoder(nn.Module):
    def __init__(self, words_dim, embedding_dim, num_layers, n_head, ff_size, drop_p):
        super(Decoder, self).__init__()
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=ff_size,
                                                 dropout=drop_p, batch_first=True, norm_first=True)
        self.pos_encoding = PositionalEncoder(embedding_dim, drop_p)
        self.emb_dropout = nn.Dropout(drop_p)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers,
                                                      norm=nn.LayerNorm(embedding_dim, eps=1e-6))
        self.words_output_layer = nn.Linear(embedding_dim, words_dim)

    def forward(self, embedded_words, encoder_output, words_future_mask, words_padding_mask, frames_padding_mask):
        embedded_words = self.pos_encoding(embedded_words)
        embedded_words = self.emb_dropout(embedded_words)
        output = self.transformer_decoder(
            tgt=embedded_words, memory=encoder_output, tgt_mask=words_future_mask,
            tgt_key_padding_mask=words_padding_mask, memory_key_padding_mask=frames_padding_mask)
        return self.words_output_layer(output).log_softmax(dim=-1)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X += self.pe[:X.size(0)]
        return self.dropout(X)
