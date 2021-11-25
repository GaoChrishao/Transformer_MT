# *_*coding:utf-8 *_*
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 采用pre_norm
class Transformer(nn.Module):

    def __init__(self, enc_vocab_size, dec_vocab_size, d_model, d_ff, num_layers, num_heads, device, enc_pdx=0,
                 dec_pdx=0, dropout=0.1):
        super(Transformer, self).__init__()

        self.device = device
        self.model_type = 'Transformer'
        self.d_model = d_model
        # self.mask_future = None

        self.enc_pdx = enc_pdx
        self.dec_pdx = dec_pdx

        self.encoder = Encoder(enc_vocab_size, d_model, num_heads, d_ff, num_layers, enc_pdx, dropout)
        self.decoder = Decoder(dec_vocab_size, d_model, num_heads, d_ff, num_layers, dec_pdx, dropout)
        self.final = nn.Linear(d_model, dec_vocab_size)

        self._model_init()

    def _model_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # return [batch_size,seq_len,seq_len]
    def future_mask(self, seq):
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        # np.ones,生成全一的方针，np.triu,生成上三角,k用于控制对角线偏移
        mask = np.triu(np.ones(attn_shape), k=1)
        mask = torch.BoolTensor(mask).to(seq.device)
        # print(mask.size())
        return mask

    # return [batch_size,1,seq_len]
    def pad_mask(self, seq_k, pad_index=0):
        pad_attn_mask = seq_k.data.eq(pad_index).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        mask = pad_attn_mask.bool()
        return mask

    def forward(self, enc_input, dec_input,enc_output=None):
        device = self.device

        # 编码器只需要进行padding mask，每个batch都不一样
        encoder_pad_mask = self.pad_mask(enc_input).to(device)

        # 解码器的自注意力层需要进行future+padding mask
        decoder_future_mask = self.future_mask(dec_input)
        decoder_pad_mask = self.pad_mask(dec_input)
        mask = decoder_pad_mask | decoder_future_mask
        decoder_self_mask = mask.to(device)

        if enc_output is None:
            enc_output = self.encoder(enc_input, encoder_pad_mask)
        dec_output = self.decoder(dec_input, decoder_self_mask, enc_output, encoder_pad_mask)
        output = self.final(dec_output)
        result = output.view(-1, output.size(-1))  # result: [batch_size x tgt_len, tgt_vocab_size]

        return enc_output,result


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, enc_pdx=0, dropout=0.1):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.enc_embedding = nn.Embedding(vocab_size, d_model, padding_idx=enc_pdx)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_inputs, mask):
        src = self.enc_embedding(enc_inputs) * math.sqrt(self.d_model)  # (batch_size,len,d_model)
        x = self.pos_encoder(src)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        out = self.norm(x)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PoswiseFeedwardNet(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)

    # x=[batch_size,seq_len,d_model]
    # mask=[batch_size,seq_len,seq_len]
    def forward(self, x, mask):
        # norm
        res, x = x, self.norm1(x)

        # self attention
        attn_out = self.attn(x, x, x, mask)  # 编码器为关键词组，需要进行padding mask
        attn_out = self.drop1(attn_out)

        # add & norm
        x = res + attn_out
        res, out1 = x, self.norm2(x)

        # ffn layer
        ffn_out = self.ffn(out1)
        ffn_out = self.drop2(ffn_out)

        # add
        out2 = res + ffn_out

        return out2


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dec_pdx=0, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dec_embedding = nn.Embedding(vocab_size, d_model, padding_idx=dec_pdx)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, dec_inputs, self_mask, enc_output, encoder_mask):
        src = self.dec_embedding(dec_inputs) * math.sqrt(self.d_model)  # (bs,len,d_model)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, self_mask, enc_output, encoder_mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.en_de_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = PoswiseFeedwardNet(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)
        self.drop3 = nn.Dropout(p=drop)

    def forward(self, x, self_mask, enc_output, enc_pad_mask):
        # norm
        res, x = x, self.norm1(x)
        # self attention
        attn_out = self.self_attn(x, x, x, self_mask)  # future mask
        attn_out = self.drop1(attn_out)
        x = res + attn_out
        # add & norm
        res, x = x, self.norm2(x)
        # encoder decoder attention
        ende_attn_out = self.en_de_attn(x, enc_output, enc_output, enc_pad_mask)  # 对编码器的输入需要进行padding mask
        ende_attn_out = self.drop2(ende_attn_out)

        # add & norm
        x = res + ende_attn_out
        res, out2 = ende_attn_out, self.norm3(x)

        # ffn layer
        ffn_out = self.ffn(out2)
        ffn_out = self.drop3(ffn_out)
        # add
        out3 = res + ffn_out

        return out3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float, requires_grad=False).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        mask: [batch_size, 1, len_q, len_v]
        '''
        d_k = Q.size(-1)
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (1.0 * np.sqrt(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_k]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.WQ = nn.Linear(d_model, self.d_k * num_heads, bias=True)
        self.WK = nn.Linear(d_model, self.d_k * num_heads, bias=True)
        self.WV = nn.Linear(d_model, self.d_v * num_heads, bias=True)
        self.fc = nn.Linear(num_heads * self.d_v, d_model, bias=True)

    def forward(self, input_Q, input_K, input_V, mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        mask: [batch_size, 1, 1,seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.WQ(input_Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Q: [batch_size, n_heads, len_q, d_k]

        K = self.WK(input_K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]

        V = self.WV(input_V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context = ScaledDotProductAttention()(Q, K, V, mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_v)

        # context: [batch_size, len_q, n_heads * d_v]
        out = self.fc(context)
        return out


class PoswiseFeedwardNet(nn.Module):
    def __init__(self, d_model, d_ff, bias=True):
        super(PoswiseFeedwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias)
        )

    def forward(self, inputs):
        return self.fc(inputs)
