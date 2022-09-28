import torch
import numpy as np
import tensorflow as tf
import re
from itertools import groupby
from Models import Vocabulary


def predict_glosses(idx_to_glosses, gloss_probabilities, frames_len, beam_size=1):
    tf.config.set_visible_devices([], 'GPU')
    gloss_probabilities = gloss_probabilities.permute(1, 0, 2).cpu().detach().numpy()
    frames_len = frames_len.cpu().detach().numpy()
    tf_gloss_probabilities = np.concatenate(
        (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
        axis=-1)
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_probabilities,
        sequence_length=frames_len,
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]

    tmp_gloss_sequences = [[] for _ in range(len(frames_len))]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1)
    predict_idx = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        predict_idx.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    predict_glosses_list = [clean_gloss_output([idx_to_glosses[idx] for idx in seq]) for seq in predict_idx]
    return predict_glosses_list


def clean_gloss_output(seq):
    seq = ' '.join(seq)
    seq = seq.strip()
    seq = re.sub(r"__LEFTHAND__", "", seq)
    seq = re.sub(r"__EPENTHESIS__", "", seq)
    seq = re.sub(r"__EMOTION__", "", seq)
    seq = re.sub(r"\b__[^_ ]*__\b", "", seq)
    seq = re.sub(r"\bloc-([^ ]*)\b", r"\1", seq)
    seq = re.sub(r"\bcl-([^ ]*)\b", r"\1", seq)
    seq = re.sub(r"\b([^ ]*)-PLUSPLUS\b", r"\1", seq)
    seq = re.sub(r"\b([A-Z][A-Z]*)RAUM\b", r"\1", seq)
    seq = re.sub(r"WIE AUSSEHEN", "WIE-AUSSEHEN", seq)
    seq = re.sub(r"^([A-Z]) ([A-Z][+ ])", r"\1+\2", seq)
    seq = re.sub(r"[ +]([A-Z]) ([A-Z]) ", r" \1+\2 ", seq)
    seq = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", seq)
    seq = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", seq)
    seq = re.sub(r"([ +][A-Z]) ([A-Z][ +])", r"\1+\2", seq)
    seq = re.sub(r"([ +]SCH) ([A-Z][ +])", r"\1+\2", seq)
    seq = re.sub(r"([ +]NN) ([A-Z][ +])", r"\1+\2", seq)
    seq = re.sub(r"([ +][A-Z]) (NN[ +])", r"\1+\2", seq)
    seq = re.sub(r"([ +][A-Z]) ([A-Z])$", r"\1+\2", seq)
    seq = re.sub(r" +", " ", seq)
    seq = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", seq)
    seq = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", seq)
    seq = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", seq)
    seq = re.sub(r"(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-])", r"\1", seq)
    seq = re.sub(r" +", " ", seq)

    # Remove white spaces and repetitions
    seq = " ".join(
        " ".join(i[0] for i in groupby(seq.split(" "))).split()
    )
    seq = seq.strip()
    return seq


def predict_words(model, frames, words, encoder_output, word_vocab, beam_size, alpha, max_seq_size=30):
    idx_to_words = word_vocab.get_itos()
    if beam_size is None:  # greedy
        predict_idx = greedy(model, frames, words, encoder_output,
                             bos_idx=word_vocab[Vocabulary.BOS_TOKEN],
                             pad_idx=word_vocab[Vocabulary.PAD_TOKEN],
                             eos_idx=word_vocab[Vocabulary.EOS_TOKEN])
    else:
        predict_idx = beam_search(model, frames, encoder_output, beam_size,
                                  sentence_length=max_seq_size,
                                  bos_idx=word_vocab[Vocabulary.BOS_TOKEN],
                                  pad_idx=word_vocab[Vocabulary.PAD_TOKEN],
                                  eos_idx=word_vocab[Vocabulary.EOS_TOKEN],
                                  alpha=alpha)

    predict_word_list = [[idx_to_words[idx] for idx in seq] for seq in predict_idx]
    return predict_word_list


def beam_search(model, frames, encoder_output, beam_size, sentence_length, bos_idx, pad_idx, eos_idx,
                alpha=1.0):
    model.eval()
    batch_size = frames.shape[0]
    # init the log prob
    topk_log_probs = torch.zeros(batch_size, beam_size, device=encoder_output.device)
    topk_log_probs[:, 1:] = -torch.inf

    # crate_the_output
    predict = torch.full([batch_size, beam_size, sentence_length], pad_idx, device=encoder_output.device)
    # first token of the first beam for in each sample is <bos>
    predict[:, :, 0] = bos_idx

    # sentences_length
    sent_ends = torch.full([batch_size, beam_size], sentence_length, device=encoder_output.device)

    frames_padding_mask = (frames.sum(dim=-1) == 0).squeeze(-1)

    for i in range(1, sentence_length):
        beam_list = []
        for j in range(beam_size):
            words = predict[:, j, :]
            words_padding_mask = (words == pad_idx)
            log_prob_bim = model.decode(words, encoder_output, frames_padding_mask, words_padding_mask)[:, i-1, :]

            # for all sent. that are finished - can predicat from now only eos, with not "cost"
            log_prob_bim[sent_ends[:, j] < sentence_length] = -1 * torch.inf
            log_prob_bim[sent_ends[:, j] < sentence_length, eos_idx] = 0
            # claculate length penalt

            sent_length = torch.minimum(sent_ends[:, j], torch.full_like(sent_ends[:, j], i))
            length_penalty = ((5.0 + (sent_length + 1)) / 6.0) ** alpha
            # adding to the prob of the rout to the probs that come out the decoder
            log_prob_bim += (topk_log_probs[:, j] / length_penalty)[:, None]
            beam_list.append(log_prob_bim)

        # flatting the beams and choose top beam_size
        log_prob = torch.cat(beam_list, dim=1)
        top_beams = torch.topk(log_prob, beam_size, dim=1)

        # for the top k beams, recreating the beam number
        beams_vec = top_beams.indices.div((log_prob.shape[1] / beam_size), rounding_mode="floor").type(torch.int64)
        # for the top k beams, recreating predcted class
        nodes_value = top_beams.indices.fmod((log_prob.shape[1] / beam_size)).type(torch.int64)
        # crating the new predicated tensor

        sent_ends = torch.cat(
            [torch.tensor([sent_ends.index_select(1, beams_vec[:, j])[k, k] for k in range(batch_size)]) for j in
             range(beam_size)]).view(
            beam_size, batch_size).T.to(encoder_output.device)

        is_end = nodes_value.eq(eos_idx)
        sent_ends[is_end] = torch.minimum(sent_ends[is_end], torch.full_like(sent_ends[is_end], i))

        predict = torch.cat([torch.cat(
            [predict.index_select(1, beams_vec[:, j])[k, k] for k in range(batch_size)])
                            .view(batch_size, sentence_length) for j in range(beam_size)]
                            ).view(beam_size, batch_size, -1).permute(1, 0, 2).to(encoder_output.device)
        predict[:, :, i] = nodes_value
        topk_log_probs = top_beams.values

    predict = [predict[i, 0, 1:min(sentence_length, sent_ends[i, 0] + 1)].tolist() for i in
               range(predict[:, 0, 1:].shape[0])]

    return predict
    # return predict[:, 0,:int(min(sent_ends[:,0].max(),sentance_length))]


def greedy(model, frames, words, encoder_output, bos_idx, eos_idx, pad_idx, max_output_length=30):
    model.eval()
    frames_padding_mask = (frames.sum(dim=-1) == 0).squeeze(-1)
    words_padding_mask = (words == pad_idx)
    batch_size = frames_padding_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_idx, dtype=torch.long)

    # a subsequent mask is intersected with this in decoder forward pass
    finished = frames_padding_mask.new_zeros(batch_size).byte()

    for _ in range(max_output_length):
        with torch.no_grad():
            logits = model.decode(
                words=ys,
                words_padding_mask=None,
                encoder_output=encoder_output,
                frames_padding_mask=frames_padding_mask
            )
            logits = logits[:, -1]
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_idx)
        finished += is_eos
        # stop predicting if <eos> reached for all elements in batch
        if (finished >= 1).sum() == batch_size:
            break
    # remove BOS-symbol
    ys = ys[:, 1:].detach().cpu().numpy()
    seq_to_list = []
    for i in range(batch_size):
        seq = []
        for word_idx in ys[i]:
            seq.append(word_idx)
            if word_idx == eos_idx:
                break
        seq_to_list.append(seq)
    return seq_to_list
