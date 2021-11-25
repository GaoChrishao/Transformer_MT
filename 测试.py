# *_*coding:utf-8 *_*
from nltk.translate.bleu_score import sentence_bleu
import math
from tool.DataTool import *
import torch.nn.functional as F
from model.Transformer import Transformer


import warnings
warnings.filterwarnings("ignore")

# 束搜索
def beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id, source_sentence, k: int):
    source_sentence = source_sentence.split(char_space)
    dec_max_len = len(source_sentence) * 1.5

    enc_input = []
    for w in source_sentence:
        enc_input.append(enc_vocab2id[w])

    dec_input = []
    dec_input.append(dec_vocab2id[char_start])

    need_search = []
    scores = []

    final_result = []
    final_scores = []

    for _ in range(k):
        need_search.append(dec_input.copy())
        scores.append(0)

    input_1 = torch.tensor([enc_input]).to(device)
    input_2 = torch.tensor([dec_input]).to(device)
    enc_output,output = model(input_1, input_2)

    # 初始化,获取概率最大的k个单词的 id
    proba = F.softmax(output[-1],dim=-1)
    idxs = proba.argsort(descending=True).view(-1)[:k]
    for i in range(k):
        max_id = idxs.data[i].item()
        need_search[i].append(max_id)
        scores[i] = math.log(proba[max_id].item())

    has_end = 0

    while has_end < k:
        temp_h = []
        temp_score = []

        for i in range(k - has_end):
            dec_input = need_search[i]
            sc = scores[i]

            # 判断该序列是否有必要继续搜索
            sentence_len = len(dec_input)
            last_word_id = dec_input[len(dec_input) - 1]
            last_word_vocab = dec_id2vocab[last_word_id]

            if last_word_vocab == char_end or sentence_len >= dec_max_len:
                has_end += 1
                final_result.append(dec_input)
                final_scores.append(sc)
                continue

            # 对该序列进行搜索
            _,output = model(torch.tensor([enc_input]).to(device), torch.tensor([dec_input]).to(device),enc_output)

            # 获取概率最大的k-hasend个单词的 id
            output = F.softmax(output[-1],dim=-1)
            idxs = output.argsort(descending=True).view(-1)[:k - has_end]

            for i in range(k - has_end):
                # print(idxs.data[i].item())
                max_id = idxs.data[i].item()
                sentence = dec_input.copy()
                sentence.append(max_id)
                temp_h.append(sentence)
                temp_score.append(sc + math.log(output[max_id].item()))

        # 如果k个句子都已经预测完成（达到最大长度或者结束符）
        if has_end == k:
            break

        # 从temp_score中选择k-hasend个最大的，放入need_search中继续搜索
        temp_score_ids = np.argsort(temp_score)[::-1]
        for i in range(k - has_end):
            max_s_id = temp_score_ids[i]
            need_search[i] = temp_h[max_s_id].copy()
            scores[i] = temp_score[max_s_id]

    return final_scores, final_result





if __name__ == '__main__':
    device = torch.device("cpu")
    print("cpu模式")

    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    print('encoder_chars:', len(encoder_chars))
    print('decoder_chars:', len(decoder_chars))
    print('max_enc_seq_length:', max_enc_seq_length)
    print('max_dec_seq_length:', max_dec_seq_length)

    enc_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    enc_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    dec_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    dec_id2vocab = {i: word for i, word in enumerate(decoder_chars)}

    print('-----------------')
    print(enc_vocab2id[char_space])
    print(dec_vocab2id[char_space])
    print('-----------------')

    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    m_state_dict = torch.load('./save/de2en_5k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
    model.load_state_dict(m_state_dict)

    model.eval()

    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    with torch.no_grad():
        # 已知序列
        test_s = open(test_file_path, 'r', encoding='utf-8').readlines()
        # 取前50条进行测试
        test_size = 872
        for line in test_s[:test_size]:
            enc_input = line.split('\t')[0]
            enc_pre_1 = enc_input.replace(" ", "")
            enc_pre_1 = enc_pre_1.replace("<e>", " ")

            target_sentence = line.split("\t")[1]
            target_sentence = target_sentence.replace(" ", "")
            target_sentence = target_sentence.replace("<e>", " ")

            print("原文：{}".format(enc_pre_1))
            print("参考译文：{}".format(target_sentence))
            k = 3
            enc_input = char_start + char_space + enc_input + char_space + char_end
            search_sources, search_result = beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id,
                                                       enc_input, k)
            for i in range(k):
                dec_input = search_result[i]
                # 将下标转化成句子
                sent = ''
                for w in dec_input:
                    sent += dec_id2vocab[w] + ' '
                sent = sent.replace(" ", "")
                sent = sent.replace("<e>", " ")
                sent = sent.replace(char_start, "")
                sent = sent.replace(char_end, "")

                bleu_score_1 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(1, 0, 0, 0))
                bleu_score_2 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 1, 0, 0))
                bleu_score_3 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 1, 0))
                bleu_score_4 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 0, 1))

                print('{:.3f},{}'.format(search_sources[i], sent))
            print(" ")
        bleu_score_1 = bleu_score_1 / test_size / k
        bleu_score_2 = bleu_score_2 / test_size / k
        bleu_score_3 = bleu_score_3 / test_size / k
        bleu_score_4 = bleu_score_4 / test_size / k
        print("bleu：{:.4f},{:.4f},{:.4f},{:.4f}".format(bleu_score_1,
                                                        bleu_score_2,
                                                        bleu_score_3,
                                                        bleu_score_4))

        print("mean bleu：{:.4f}".format((bleu_score_1 + bleu_score_2 + bleu_score_3 + bleu_score_4) / 4))
