import torch
import torch.nn as nn
import math
from Models.Embedder import AlexNetSE
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class SLTModel(nn.Module):
    @staticmethod
    def get_mask_for_seq(seq_len, device):
        mask = torch.ones(seq_len, seq_len)
        if device >= 0:
            mask = mask.to(device)
        return torch.triu(mask, diagonal=1)

    def __init__(self, gloss_dim, words_dim, embedding_dim=1024, num_layers_encoder=2, num_layers_decoder=2, n_head=8,
                 ff_size=2048, drop_p=0.1, spatial_flag=False):
        super(SLTModel, self).__init__()
        self.spatial_flag = spatial_flag
        self.embedding_dim = embedding_dim
        self.spatial_embedding = AlexNetSE(drop_p) if spatial_flag else None
        self.encoder = Encoder(embedding_dim, num_layers_encoder, n_head, ff_size, drop_p)
        self.decoder = Decoder(words_dim, embedding_dim, num_layers_decoder, n_head, ff_size, drop_p)
        self.gloss_output_layer = nn.Linear(embedding_dim, gloss_dim)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.gloss_output_layer.bias.data.zero_()
        self.gloss_output_layer.weight.data.uniform_(-init_range, init_range)

    def forward(self, frames, words):
        if self.spatial_embedding:
            embedded_frames = self.spatial_embedding(frames) * math.sqrt(self.embedding_dim)
        else:
            embedded_frames = frames.squeeze(dim=2).clone()
        encoder_output = self.encoder(embedded_frames)
        glosses_prob_output = self.gloss_output_layer(encoder_output).log_softmax(dim=-1)
        frames_mask = self.get_mask_for_seq(encoder_output.shape[1], encoder_output.get_device())
        words_mask = self.get_mask_for_seq(words.shape[1], words.get_device())
        decoder_output = self.decoder(words, encoder_output, words_mask, frames_mask)
        return decoder_output, glosses_prob_output


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, n_head, ff_size, drop_p):
        super(Encoder, self).__init__()
        self.emb_dim = embedding_dim
        # self.pos_encoder = PositionalEncoder(embedding_dim, drop_p)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=ff_size,
                                                 dropout=drop_p, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, X):
        return self.transformer_encoder(X)


class Decoder(nn.Module):
    def __init__(self, words_dim, embedding_dim, num_layers, n_head, ff_size, drop_p):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(words_dim, embedding_dim)
        decoder_layers = TransformerDecoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=ff_size,
                                                 dropout=drop_p, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        self.words_output_layer = nn.Linear(embedding_dim, words_dim)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.word_embedding.weight.data.uniform_(-init_range, init_range)
        self.words_output_layer.bias.data.zero_()
        self.words_output_layer.weight.data.uniform_(-init_range, init_range)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # TODO: check if need to send memory_mask
        word_embedding = self.word_embedding(tgt)
        output = self.transformer_decoder(word_embedding, memory, tgt_mask)
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

