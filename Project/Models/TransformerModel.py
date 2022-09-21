import torch
import torch.nn as nn
import math
from Models.Embedding import AlexNetSE, FrameEmbedding, WordEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class SLTModel(nn.Module):
    @staticmethod
    def get_future_mask_for_seq(seq_len, device):
        mask = torch.ones(seq_len, seq_len)
        if device >= 0:
            mask = mask.to(device)
        return torch.triu(mask, diagonal=1).type(torch.bool)

    def __init__(self, frame_size, gloss_dim, words_dim, word_padding_idx, emo_dim = 0, embedding_dim=512, num_layers_encoder=2, num_layers_emo =2,
                 num_layers_decoder=2, n_head=8, ff_size=2048, drop_p=0.1, spatial_flag=False):
        super(SLTModel, self).__init__()
        self.spatial_flag = spatial_flag
        self.embedding_dim = embedding_dim
        self.emo_dim = emo_dim
        self.frame_embedding = AlexNetSE() if spatial_flag else FrameEmbedding(input_size=frame_size,
                                                                               emb_dim=embedding_dim)
        self.word_padding_index = word_padding_idx
        self.word_embedding = WordEmbedding(padding_idx=word_padding_idx, input_size=words_dim, emb_dim=embedding_dim)
        self.encoder = Encoder(embedding_dim, num_layers_encoder, n_head, ff_size, drop_p)
        self.decoder = Decoder(words_dim, embedding_dim, num_layers_decoder, n_head, ff_size, drop_p)
        self.gloss_output_layer = nn.Linear(embedding_dim, gloss_dim)
        if emo_dim > 0:
            self.encoder_emo = Encoder(embedding_dim, num_layers_emo, n_head, ff_size, drop_p)
            self.merger = Merger(embedding_dim,2)
            self.emo_output_layer= nn.Linear(embedding_dim, emo_dim)

        self.init_weights()

    def init_weights(self):
        init_gain = 1.0
        for m in self.modules():
            if hasattr(m, 'bias'):
                m.bias.data.zero_()
            if hasattr(m, 'weight') and len(m.weight.shape) > 1:
                nn.init.xavier_uniform_(m.weight, gain=init_gain)

    def encode(self, frames, frames_padding_mask):
        if self.spatial_flag:
            embedded_frames = self.spatial_embedding(frames) * math.sqrt(self.embedding_dim)
        else:
            embedded_frames = self.frame_embedding(frames.squeeze(dim=2), frames_padding_mask)
        if self.emo_dim > 0:
            return self.encoder(embedded_frames, frames_padding_mask) , self.encoder_emo(embedded_frames, frames_padding_mask)


        return self.encoder(embedded_frames, frames_padding_mask)

    def decode(self, words, encoder_output,  frames_padding_mask, words_padding_mask):
        words_future_mask = self.get_future_mask_for_seq(words.shape[1], words.get_device())
        embedded_words = self.word_embedding(words, words_padding_mask)
        decoder_output = self.decoder(embedded_words, encoder_output, words_future_mask,
                                      words_padding_mask, frames_padding_mask)
        return decoder_output

    def forward(self, frames, words):
        frames_padding_mask = (frames.sum(dim=-1) == 0).squeeze(-1)
        words_padding_mask = (words == self.word_padding_index)
       
        if self.emo_dim > 0:
            encoder_output , emo_output = self.encode(frames, frames_padding_mask)
            glosses_prob_output = self.gloss_output_layer(encoder_output).log_softmax(dim=-1)
            emo_prob_output = self.emo_output_layer(emo_output).log_softmax(dim=-1)
            merge_output  = self.merger(encoder_output, emo_output)#.to(words.get_device())
            decoder_output = self.decode(words, merge_output, frames_padding_mask, words_padding_mask)
            return decoder_output, glosses_prob_output, encoder_output, emo_prob_output
       
        encoder_output  = self.encode(frames, frames_padding_mask)
        glosses_prob_output = self.gloss_output_layer(encoder_output).log_softmax(dim=-1)
        decoder_output = self.decode(words, encoder_output, frames_padding_mask, words_padding_mask)
        return decoder_output, glosses_prob_output, encoder_output


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, n_head, ff_size, drop_p):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.emb_dropout = nn.Dropout(drop_p)
        self.pos_encoder = PositionalEncoder(embedding_dim, drop_p)
        encoder_layers = TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dim_feedforward=ff_size,
                                                 dropout=drop_p, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers,
                                                      norm=nn.LayerNorm(embedding_dim, eps=1e-6))

    def forward(self, embedded_frames, frames_padding_mask):
        output = self.pos_encoder(embedded_frames)
        output = self.emb_dropout(output)
        return self.transformer_encoder(output, src_key_padding_mask=frames_padding_mask)

class Merger(nn.Module):
    def __init__(self, dim, num_layers=1):
        super(Merger, self).__init__()
        self.num_layers = num_layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=2 * dim, out_features=dim),
            nn.ReLU()
        )
        self.layers_list =  nn.ModuleList([nn.Sequential(
            nn.Linear(in_features= dim, out_features=dim),
            nn.ReLU()
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
                                                 dropout=drop_p, batch_first=True)
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

