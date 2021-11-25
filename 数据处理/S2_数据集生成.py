import random
import string

from tool.Global import *
from tool.DataTool import load_bpe_dic
import os

os.chdir("../")
random.seed(0)


# 利用BEP得到的结果进行单词划分，贪心算法
def bpe_convert(bpe, word):
    # print(word+"----",end='')
    word = word + word_end
    result = []
    while True:
        find = False
        for bpe_item in bpe:
            if word.startswith(bpe_item):
                result.append(bpe_item)
                word = word[len(bpe_item):]
                find = True
                break
        if not find:
            result.append(char_unknown)
            break
        if word is None or len(word) == 0:
            break

    return result


def create_vocab(encoder_file_path,
                 decoder_file_path,
                 encoder_bpe_path,
                 decoder_bpe_path,
                 file_output_path):
    encoder_bpe_result, decoder_bpe_result = load_bpe_dic(encoder_bpe_path, decoder_bpe_path, sort_by_length=True)

    encoder_lines = open(encoder_file_path, 'r', encoding='utf-8').readlines()
    decoder_lines = open(decoder_file_path, 'r', encoding='utf-8').readlines()
    if len(encoder_lines) != len(decoder_lines):
        print("{},{}数据集长度不一致！！！".format(encoder_file_path, decoder_file_path))
        return 0

    print("总数据量：{}".format(len(encoder_lines)))

    with open(file_output_path, 'w', encoding='utf-8') as out:
        for i in range(0, len(encoder_lines)):
            line_enc = encoder_lines[i].strip()
            line_dec = decoder_lines[i].strip()

            enc_words = line_enc.split(char_space)
            enc_words_bpe = []
            for item in enc_words:
                ##将单词切分
                words_bpe = bpe_convert(encoder_bpe_result, item)
                for words_son_part in words_bpe:
                    enc_words_bpe.append(words_son_part)

            dec_words = line_dec.split(char_space)
            dec_words_bpe = []
            for item in dec_words:
                ##将单词切分
                words_bpe = bpe_convert(decoder_bpe_result, item)
                for words_son_part in words_bpe:
                    dec_words_bpe.append(words_son_part)

            keysline = ''.join(["%s " % i for i in enc_words_bpe]).strip()
            out.write(keysline)
            out.write("\t")

            keysline = ''.join(["%s " % i for i in dec_words_bpe]).strip()
            out.write(keysline)
            out.write("\n")

            print("\r%.2f %%" % ((i + 1) * 100.0 / len(encoder_lines)), end="")

    return count


def create_train_test(input_file_path,
                      train_file_path,
                      test_file_path,
                      valid_file_path,
                      all_samples,
                      ration=0.02):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        f_train = open(train_file_path, 'w', encoding='utf-8')
        f_test = open(test_file_path, 'w', encoding='utf-8')
        f_valid = open(valid_file_path, 'w', encoding='utf-8')
        count = 0
        seperated_num = (int)(all_samples * ration)
        divide = int(1 / ration)
        print("验证集和测试集所占比例：{}，数量：{}".format(ration, seperated_num))
        line = input_file.readline()
        while line:
            if (count + 1) % divide == 0:
                if random.random() < 0.5:
                    f_valid.write(line)
                else:
                    f_test.write(line)
            else:
                f_train.write(line)
            count += 1
            line = input_file.readline()
        f_train.close()
        f_valid.close()
        f_test.close()

    return count


if __name__ == '__main__':

    # 将原始的数据集合并，并使用bpe词典进行分析
    count = create_vocab(
        corpus_encoder_path,
        corpus_decoder_path,
        encoder_bpe_dic_path,
        decoder_bpe_dic_path,
        combined_vocab_path)

    if count > 0:
        print("\n实际处理的样本数量:%d" % (count))

        # 划分训练集和测试集
        create_train_test(
            combined_vocab_path,
            train_file_path,
            test_file_path,
            valid_file_path,
            count,
            0.01
        )
        print("划分训练集、测试集、验证集")
