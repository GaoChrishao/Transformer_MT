# *_*coding:utf-8 *_*
import random

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch as torch

import json
import os


from tool.Global import *


def load_bpe_dic(encoder_bpe_path, decoder_bpe_path, sort_by_length=False):
    encoder_bpe_result = []
    decoder_bpe_result = []
    with open(encoder_bpe_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            # 可能出现这种情况      ":<e>:21"
            tmp = line.split(":")
            if len(tmp) > 2:
                left_part = ''.join(tmp[:-1])
            else:
                left_part = tmp[0]
            encoder_bpe_result.append(left_part)
            line = f.readline().strip()

    with open(decoder_bpe_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            # 可能出现这种情况      ":<e>:21"
            tmp = line.split(":")
            if len(tmp) > 2:
                left_part = ''.join(tmp[:-1])
            else:
                left_part = tmp[0]
            decoder_bpe_result.append(left_part)
            line = f.readline().strip()

    if sort_by_length:
        encoder_bpe_result.sort(key=lambda x: len(x), reverse=True)
        decoder_bpe_result.sort(key=lambda x: len(x), reverse=True)
    return encoder_bpe_result, decoder_bpe_result


def _calculate(filename):
    max_encoder_length = 0
    max_decoder_length = 0
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            tmp = line.split("\t")
            encoder_sentence = tmp[0]
            decoder_sentence = tmp[1]

            encoder_words = encoder_sentence.split(char_space)
            length_line = len(encoder_words)
            if length_line > max_encoder_length:
                max_encoder_length = length_line

            decoder_words = decoder_sentence.split(char_space)
            length_line = len(decoder_words)
            if length_line > max_decoder_length:
                max_decoder_length = length_line

            line = f.readline().strip()
    return max_encoder_length, max_decoder_length


def calculate_data():
    if os.path.exists(data_path_vocab_desc):
        with open(data_path_vocab_desc, 'r', encoding='utf-8') as json_file:
            dic = json.load(json_file)
            max_enc_seq_length = dic['max_enc_seq_length']
            max_dec_seq_length = dic['max_dec_seq_length']
            encoder_chars = dic['encoder_chars']
            decoder_chars = dic['decoder_chars']
        print("载入已有字典")
        return encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length

    max_enc_seq_length, max_dec_seq_length = _calculate(combined_vocab_path)

    encoder_chars, decoder_chars = load_bpe_dic(encoder_bpe_dic_path, decoder_bpe_dic_path)

    # 插入特俗字符
    encoder_chars.insert(0, char_space)
    encoder_chars.insert(1, char_start)
    encoder_chars.insert(2, char_end)
    encoder_chars.insert(3, char_unknown)

    decoder_chars.insert(0, char_space)
    decoder_chars.insert(1, char_start)
    decoder_chars.insert(2, char_end)
    decoder_chars.insert(3, char_unknown)

    with open(data_path_vocab_desc, 'w', encoding='utf-8') as json_file:
        dic = {}
        dic['max_enc_seq_length'] = max_enc_seq_length
        dic['max_dec_seq_length'] = max_dec_seq_length
        dic['encoder_chars'] = encoder_chars
        dic['decoder_chars'] = decoder_chars
        json.dump(dic, json_file, ensure_ascii=False)
        print("保存字典")

    return encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length

def load_dataset(file_path):
    input_file = open(file_path, 'r', encoding='utf-8')
    line = input_file.readline().strip()
    sorted_pair = []
    while line:
        sorted_pair.append(line)
        line = input_file.readline().strip()

    input_file.close()
    sorted_pair.sort(key=lambda x: len(x))

    num_samples = len(sorted_pair)
    print("数据集大小:{}".format(num_samples))
    return num_samples,sorted_pair

def data_generator(
        source_vocab2id,
        target_vocab2id,
        data_sorted_pair,
        _default_batch_size,  # 默认batch_size，需要先进行测试，看GPU显存是否够
        default_max_encoder_length,  # 整个训练集中最长的句子长度，不添加<start>和<end>
        default_max_decoder_length,
        use_auto_batch_size=True  # 开启该选项后，会自动根据每个batch的最大长度调整batch_size大小,更换设备和模型等参数后，需要手动调整
):
    num_samples = len(data_sorted_pair)
    index = 0

    # 记录了上一个batch_size中，encoder和decoder的最大长度,用于对batch_size进行估计
    enc_max_len = 0
    dec_max_len = 0
    all_batches=[]
    while index<num_samples:

        enc_batch_data = []
        dec_batch_data = []

        if use_auto_batch_size:
            if index == 0:
                batch_size = 200
            else:

                batch_size = int((default_max_encoder_length+default_max_decoder_length) * _default_batch_size / (enc_max_len + dec_max_len)) #此处需要根据自己的配置进行调整
            # print("\n{}\n".format(batch_size))
            enc_max_len = 0
            dec_max_len = 0
        else:
            batch_size = _default_batch_size
            enc_max_len = default_max_encoder_length
            dec_max_len = default_max_decoder_length

        for i in range(0, batch_size):
            line = data_sorted_pair[(index + i) % num_samples] #最后一个batch时，会取部分前面的数据
            batch_index = line.split("\t")
            s_source = batch_index[0].strip()
            s_target = batch_index[1].strip()

            s_source = char_start + char_space + s_source + char_space + char_end
            s_target = char_start + char_space + s_target + char_space + char_end

            s_source = s_source.split(char_space)
            s_target = s_target.split(char_space)
            enc_max_len = len(s_source) if enc_max_len < len(s_source) else enc_max_len
            dec_max_len = len(s_target) if dec_max_len < len(s_target) else dec_max_len
            enc_batch_data.append(s_source)
            dec_batch_data.append(s_target)
        # print("{},{}".format(enc_max_len,dec_max_len))

        data_enc = np.zeros(
            (batch_size, enc_max_len),  # 编码器的输入需要加上<start> <end>两个字符
            dtype='int32')
        data_dec = np.zeros(
            (batch_size, dec_max_len),  # 解码器的输入需要加上<start> <end>两个字符
            dtype='int32')

        # batch_index：一个batch内部的句子的顺序
        for batch_index in range(0, batch_size):
            enc_line = enc_batch_data[batch_index]
            dec_line = dec_batch_data[batch_index]

            for t in range(len(enc_line)):
                if enc_line[t] in source_vocab2id:
                    data_enc[batch_index, t] = source_vocab2id[enc_line[t]]
                else:
                    data_enc[batch_index, t] = source_vocab2id[char_unknown]
            data_enc[batch_index, t + 1:] = source_vocab2id[char_space]

            for t in range(len(dec_line)):
                if dec_line[t] in target_vocab2id:
                    data_dec[batch_index, t] = target_vocab2id[dec_line[t]]
                else:
                    data_dec[batch_index, t] = target_vocab2id[char_unknown]

            data_dec[batch_index, t + 1:] = target_vocab2id[char_space]


        dec_input = data_dec[:, 0:dec_max_len - 1]
        dec_output = data_dec[:, 1:dec_max_len]
        all_batches.append([data_enc,dec_input,dec_output])

        index += batch_size

    batch_num=len(all_batches)
    print("batch num:{}".format(batch_num))

    while True:
        random.shuffle(all_batches)
        for i in range(0,batch_num):
            enc_input = torch.LongTensor(all_batches[i][0])
            dec_input = torch.LongTensor(all_batches[i][1])
            dec_output = torch.LongTensor(all_batches[i][2])
            yield batch_num, enc_input, dec_input, dec_output
        yield batch_num,None,None,None
