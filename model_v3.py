
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils_pg import *

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, encode_outputs_size, embedding_size, dict_size, device):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.encode_outputs_size = encode_outputs_size
        self.embedding_size = embedding_size
        self.dict_size = dict_size
        self.device = device

        self.V1 = nn.Linear(self.hidden_size + self.encode_outputs_size + self.embedding_size, self.hidden_size)
        self.V2 = nn.Linear(self.hidden_size, self.dict_size)
        self.W = nn.Linear(self.hidden_size + self.encode_outputs_size + self.embedding_size, 1)

    def forward(self, decode_hidden_states, context_vector, embedded_y, att_dist=None, xids=None, max_ext_len=None):
        h = torch.cat((decode_hidden_states, context_vector, embedded_y), 2)
        logit = self.V2(torch.tanh(self.V1(h)))
        y_dec = torch.softmax(logit, dim=2)

        ext_zeros = torch.zeros(y_dec.size(0), y_dec.size(1), max_ext_len).to(self.device)
        y_dec = torch.cat((y_dec, ext_zeros), 2)
        g = torch.sigmoid(self.W(h))
        y_dec = (g * y_dec).scatter_add(2, xids, (1 - g) * att_dist)

        return y_dec


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, encode_outputs_size, device, is_predicting):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encode_outputs_size = encode_outputs_size
        self.is_predicting = is_predicting
        self.device = device

        self.W_h = nn.Linear(self.encode_outputs_size, self.encode_outputs_size, bias=False)
        self.W_s = nn.Linear(self.hidden_size, self.encode_outputs_size, bias=False)
        self.W_c = nn.Linear(1, self.encode_outputs_size, bias=False)
        self.b_attn = nn.Parameter(torch.Tensor(self.encode_outputs_size))  # 初始化
        self.V = nn.Linear(self.encode_outputs_size, 1, bias=False)

        self.gru1 = nn.GRU(self.input_size, self.hidden_size)
        self.gru2 = nn.GRU(2 * self.hidden_size, self.hidden_size)


    def attention(self, encode_outputs, s1, x_mask, coverage_vector=None):

        coverage_t = torch.transpose(coverage_vector, 0, 1).unsqueeze(2)
        # W_h * h_i + W_s * s_t + W_c * c_t + b_attn
        attention_states = self.W_h(encode_outputs) + self.W_s(s1) + self.W_c(coverage_t) + self.b_attn
        attention_score = self.V(torch.tanh(attention_states) * x_mask)
        attention_weights = attention_score.masked_fill(x_mask == 0, -1e9).softmax(0)

        return attention_weights

    def decode_one_step(self, y, y_mask, hidden, encode_outputs, x_mask, coverage_vector=None):
        s1 = self.gru1(y.unsqueeze(0), hidden.unsqueeze(0))[1].squeeze(0)
        s1 = y_mask * s1 + (1.0 - y_mask) * hidden

        attention_weights = self.attention(encode_outputs, s1, x_mask, coverage_vector)
        context_vector = torch.sum(attention_weights * encode_outputs, 0)

        s2 = self.gru2(context_vector.unsqueeze(0), s1.unsqueeze(0))[1].squeeze(0)
        s2 = y_mask * s2 + (1.0 - y_mask) * s1

        words_attention_weight = torch.transpose(attention_weights.reshape(x_mask.size(0), -1), 0, 1)

        coverage_vector += words_attention_weight
        return s2, context_vector, words_attention_weight, coverage_vector




    def forward(self, embedded_y, encode_outputs, init_state, x_mask, y_mask, x_index=None, init_coverage=None):
        decode_seq_len = embedded_y.size()[0]
        hidden = init_state
        x_index = torch.transpose(x_index, 0, 1)  # [batch_size, enc_seq_len]
        coverage_vector = init_coverage

        hidden_states = torch.zeros((decode_seq_len, *hidden.size()), device=self.device)
        context_vecotors = torch.zeros((decode_seq_len, *encode_outputs[0, :, :].size()), device=self.device)
        words_attention_weights = torch.zeros((decode_seq_len, *x_index.size()), device=self.device)
        coverage_vectors_all = torch.zeros((decode_seq_len + 1, *coverage_vector.size()), device=self.device)
        x_indexes = torch.zeros((decode_seq_len, *x_index.size()), device=self.device, dtype=torch.int64)


        for t in range(decode_seq_len):

            coverage_vectors_all[t, :, :] = coverage_vector
            hidden, att, att_dist, coverage_vector = self.decode_one_step(embedded_y[t], y_mask[t], hidden, encode_outputs, x_mask,
                                                           coverage_vector)
            hidden_states[t, :, :] = hidden
            context_vecotors[t, :, :] = att
            words_attention_weights[t, :, :] = att_dist
            x_indexes[t, :, :] = x_index


        if self.is_predicting:

            coverage_vectors_all[-1, :, :] = coverage_vector
            coverage_vectors = coverage_vectors_all[1:, :, :]
        else:
            coverage_vectors = coverage_vectors_all[:-1, :, :]


        return hidden_states, context_vecotors, words_attention_weights, x_indexes, coverage_vectors


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        self.cfg = cfg
        # self.is_predicting = cfg["is_predicting"]
        # self.is_bidirectional = cfg["is_bidirectional"]
        # self.beam_decoding = cfg["beam_decoding"]
        # self.cell = cfg["cell"]
        self.device = self.cfg.device
        # self.copy = cfg["copy"]
        # self.coverage = cfg["coverage"]
        # self.avg_nll = cfg["avg_nll"]

        self.embedding_size = self.cfg.embedding_size
        # self.len_x = cfg["len_x"]
        # self.len_y = cfg["len_y"]
        self.hidden_size = self.cfg.hidden_size
        self.dict_size = self.cfg.dict_size
        self.pad_token_idx = self.cfg.pad_token_idx
        self.ctx_size = self.hidden_size * 2

        self.embeddings = nn.Embedding(self.dict_size, self.embedding_size, self.pad_token_idx)
        self.encoder = nn.GRU(self.embedding_size, self.hidden_size, bidirectional=True)
        self.decoder = Decoder(self.embedding_size, self.hidden_size, self.ctx_size, self.device, self.cfg.is_predicting)

        self.decode_apdapter = nn.Linear(self.ctx_size, self.hidden_size)
        self.word_prob = WordProbLayer(self.hidden_size, self.ctx_size, self.embedding_size, self.dict_size, self.device)
        self.loss = torch.nn.NLLLoss(ignore_index=0)

    def encode(self, input_x, len_x):
        self.encoder.flatten_parameters()
        embedded_x = self.embeddings(input_x)

        embedded_x = pack_padded_sequence(embedded_x, len_x)
        encode_outputs, encode_hidden_states = self.encoder(embedded_x)
        encode_outputs, _ = pad_packed_sequence(encode_outputs)

        dec = encode_hidden_states.permute(1, 0, 2).reshape(encode_hidden_states.size()[1], -1)
        dec_init_state = torch.tanh(self.decode_apdapter(dec))
        return encode_outputs, dec_init_state

    def decode_once(self, y, encode_outputs, dec_init_state, mask_x, x=None, max_ext_len=None, coverage_vector=None):
        batch_size = encode_outputs.size(1)
        if torch.sum(y) < 0:
            embedded_y = torch.zeros((1, batch_size, self.embedding_size)).to(self.device)
        else:
            embedded_y = self.embeddings(y)
        mask_y = torch.ones((1, batch_size, 1)).to(self.device)

        dec_status, atted_context, att_dist, xids, C = self.decoder(embedded_y, encode_outputs, dec_init_state, mask_x, mask_y,
                                                                         x, coverage_vector)

        y_pred = self.word_prob(dec_status, atted_context, embedded_y, att_dist, xids, max_ext_len)

        return y_pred, dec_status, C


    def forward(self, input_x, len_x, y, mask_x, mask_y, x_ext, y_ext, max_ext_len):

        encode_outputs, dec_init_state = self.encode(input_x, len_x)

        embedded_y = self.embeddings(y)
        y_shifted = embedded_y[:-1, :, :]
        y_shifted = torch.cat((torch.zeros(1, *y_shifted[0].size()).to(self.device), y_shifted), 0)

        coverage_vector = torch.zeros(torch.transpose(input_x, 0, 1).size()).to(self.device)

        dec_status, atted_context, att_dist, xids, C = self.decoder(y_shifted, encode_outputs, dec_init_state, mask_x, mask_y, x_ext,
                                                                         coverage_vector)

        y_pred = self.word_prob(dec_status, atted_context, y_shifted, att_dist, xids, max_ext_len)

        y_pred_log = torch.log(y_pred + 1e-12)
        cost = self.loss(y_pred_log.reshape(-1, y_pred.size()[-1]), y.reshape(-1))


        cost_c = torch.mean(torch.sum(torch.min(att_dist, C), 2))
        return y_pred, cost, cost_c


