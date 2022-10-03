import torch
import torch.nn as nn
import math

from Models.TransformerModel import Encoder, Decoder, Merger
from Models.Embedding import AlexNetSE, FrameEmbedding, WordEmbedding


class SLTModel(nn.Module):
    @staticmethod
    def get_future_mask_for_seq(seq_len, device):
        mask = torch.ones(seq_len, seq_len)
        if device >= 0:
            mask = mask.to(device)
        return torch.triu(mask, diagonal=1).type(torch.bool)

    def __init__(self, frame_size, gloss_dim, words_dim, word_padding_idx, embedding_dim=512,
                 num_layers_encoder=3, num_layers_decoder=3, n_head=8, ff_size=2048, dropout_encoder=0.1,
                 dropout_decoder=0.1, spatial_flag=False):
        super(SLTModel, self).__init__()
        self.spatial_flag = spatial_flag
        self.embedding_dim = embedding_dim
        self.frame_embedding = AlexNetSE() if spatial_flag else FrameEmbedding(input_size=frame_size,
                                                                               emb_dim=embedding_dim)
        self.word_padding_index = word_padding_idx
        self.word_embedding = WordEmbedding(padding_idx=word_padding_idx, input_size=words_dim, emb_dim=embedding_dim)
        self.encoder = Encoder(embedding_dim, num_layers_encoder, n_head, ff_size, dropout_encoder)
        self.decoder = Decoder(words_dim, embedding_dim, num_layers_decoder, n_head, ff_size, dropout_decoder)
        self.gloss_output_layer = nn.Linear(embedding_dim, gloss_dim)
        self.init_weights()

    def init_weights(self):
        init_gain = 1.0
        with torch.no_grad():
            for name, p in self.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(p)
                    # m.bias.data.zero_()
                elif len(p.size()) > 1:
                    nn.init.xavier_uniform_(p.data, gain=init_gain)
            # zero out paddings
            self.word_embedding.embedding.weight.data[self.word_padding_index].zero_()

    def encode(self, frames, frames_padding_mask):
        if self.spatial_flag:
            embedded_frames = self.spatial_embedding(frames) * math.sqrt(self.embedding_dim)
        else:
            embedded_frames = self.frame_embedding(frames.squeeze(dim=2), frames_padding_mask)
        return self.encoder(embedded_frames, frames_padding_mask)

    def decode(self, words, encoder_output, frames_padding_mask, words_padding_mask):
        words_future_mask = self.get_future_mask_for_seq(words.shape[1], words.get_device())
        embedded_words = self.word_embedding(words, words_padding_mask)
        decoder_output = self.decoder(embedded_words, encoder_output, words_future_mask,
                                      words_padding_mask, frames_padding_mask)
        return decoder_output

    def forward(self, frames, words):
        frames_padding_mask = (frames.sum(dim=-1) == 0).squeeze(-1)
        words_padding_mask = (words == self.word_padding_index)
        encoder_output = self.encode(frames, frames_padding_mask)
        glosses_scores_output = self.gloss_output_layer(encoder_output)
        glosses_probs_outputs = glosses_scores_output.log_softmax(dim=-1)
        decoder_output = self.decode(words, encoder_output, frames_padding_mask, words_padding_mask)
        return decoder_output, glosses_probs_outputs, glosses_scores_output, encoder_output


class SLTEmotionsModel(SLTModel):
    def __init__(self, frame_size, gloss_dim, words_dim, word_padding_idx, emo_dim=0, embedding_dim=512,
                 num_layers_encoder=2, num_layers_emo=2, num_layers_decoder=2, n_head=8, ff_size=2048,
                 dropout_encoder=0.1, dropout_decoder=0.1, spatial_flag=False):
        super().__init__(frame_size, gloss_dim, words_dim, word_padding_idx, embedding_dim,
                         num_layers_encoder, num_layers_decoder, n_head, ff_size,
                         dropout_encoder, dropout_decoder, spatial_flag)
        self.emo_dim = emo_dim
        self.emotions_encoder = Encoder(embedding_dim, num_layers_emo, n_head, ff_size, dropout_encoder)
        self.merger = Merger(embedding_dim, dropout_encoder, 2)
        self.emo_output_layer = nn.Linear(embedding_dim, emo_dim)
        self.init_weights()

    def encode(self, frames, frames_padding_mask):
        if self.spatial_flag:
            embedded_frames = self.spatial_embedding(frames) * math.sqrt(self.embedding_dim)
        else:
            embedded_frames = self.frame_embedding(frames.squeeze(dim=2), frames_padding_mask)
        glosses_encoder_output = self.encoder(embedded_frames, frames_padding_mask)
        emotions_encoder_output = self.emotions_encoder(embedded_frames, frames_padding_mask)
        return glosses_encoder_output, emotions_encoder_output

    def forward(self, frames, words):
        frames_padding_mask = (frames.sum(dim=-1) == 0).squeeze(-1)
        words_padding_mask = (words == self.word_padding_index)

        encoder_output, emo_output = self.encode(frames, frames_padding_mask)
        glosses_prob_output = self.gloss_output_layer(encoder_output).log_softmax(dim=-1)
        emo_prob_output = self.emo_output_layer(emo_output).log_softmax(dim=-1)
        merge_output = self.merger(encoder_output, emo_output)  # .to(words.get_device())
        # merge_output =  torch.concat([encoder_output,emo_output],dim=-1).to(encoder_output.device)
        decoder_output = self.decode(words, merge_output, frames_padding_mask, words_padding_mask)
        return decoder_output, glosses_prob_output, encoder_output, emo_prob_output, merge_output

