# *_*coding:utf-8 *_*
import re
import jieba
import unicodedata
from tool import Global
from tool.Global import char_end, bpe_symbol


class BPE:
    def __init__(self, size=1500):
        self.size = size

    def loadData(self, file_path):
        result = []
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            while line:
                line = line.strip()
                words = line.split(" ")
                for w in words:
                    result.append(w)
                line = f.readline()
                count += 1
        print("读取{}行数据.".format(count))
        return result

    def apperInStr(self, part, word):
        repeat = 0
        if word == part:
            repeat = 1
        elif word.startswith(part + " "):
            repeat = 1
        elif word.endswith(" " + part):
            repeat = 1
        else:
            repeat = word.count(" " + part + " ")
        return repeat

    # bpe算法的主体部分
    # input=["who","are","you","?","net","sentence"......]
    def split(self, input: [], output_path):
        # input: 句子分词后的结果

        # 1.按字符拆分，并添加终结符<e>，统计词频
        corpus = {}
        bpe_result = {Global.word_end: 0}
        print("载入数据")

        for word in input:
            # 统计词频
            # result初始化为各个基础字符的统计，例如:["a":4,"b":2,"c":2,"<e>":4]
            for c in word:
                if c in bpe_result:
                    bpe_result[c] += 1
                else:
                    bpe_result[c] = 1
            bpe_result[Global.word_end] += 1

            # 拆分并添加终结符
            word_split = " ".join(word)
            word_split = word_split + " " + Global.word_end
            if word_split not in corpus:
                corpus[word_split] = 1
            else:
                corpus[word_split] += 1

        # corpus={'y o u <e>': 1, 'a r e <e>': 1, 'w h o <e>': 2}
        # result={'u', 'y', 'a', 'w', 'e', '<e>', 'r', 'h', 'o'}
        print("初始化完毕")

        while len(bpe_result) < self.size:

            # 在语料库中统计每两个连续的子串共同出现的次数
            # gram_2={'w h':2, 'h o':2, 'o <e>':2, 'y o':1, 'o u':1, 'u <e>'1, 'a r':1, 'r e':1, 'e <e>':1}
            # 每次查找出现最多的进行合并,可以采用最大堆进行优化
            gram_2 = {}
            for word in corpus:
                chars = word.split(" ")
                for i in range(0, len(chars) - 1):
                    char_2 = chars[i] + " " + chars[i + 1]
                    if char_2 in gram_2:
                        gram_2[char_2] += corpus[word]  # 添加时，需要根据在语料库中出现次数进行添加
                    else:
                        gram_2[char_2] = corpus[word]

            # 选取出现频率最高的2-gram进行合并
            max_freq_gram_2 = ""
            max_freq = 0
            for word in gram_2:
                freq = gram_2[word]
                if max_freq < freq:
                    max_freq = freq
                    max_freq_gram_2 = word

            if max_freq == 1:
                print("max_freq==1,停止合并")
                break

            if len(gram_2) == 0:
                print("没有可以合并的单词，停止合并")
                break

            # 对语料库中中数据进行合并
            max_gram_combined = max_freq_gram_2.replace(" ", "")
            bpe_result[max_gram_combined] = gram_2[max_freq_gram_2]

            # 对corpus中选出的这个gram_2合并为一个
            new_corpus = {}
            for word in corpus:
                freq = corpus[word]
                new_word = word
                if word == max_freq_gram_2:
                    new_word = max_gram_combined
                elif word.startswith(max_freq_gram_2 + " "):
                    new_word = word.replace(max_freq_gram_2 + " ", max_gram_combined + " ")
                elif word.endswith(" " + max_freq_gram_2):
                    new_word = word.replace(" " + max_freq_gram_2, " " + max_gram_combined)
                elif " " + max_freq_gram_2 + " " in word:
                    new_word = word.replace(" " + max_freq_gram_2 + " ", " " + max_gram_combined + " ")
                new_corpus[new_word] = freq

            corpus = new_corpus
            # print("corpus:", corpus)

            # 计算合并后的a,b在新的语料库中单独出现的次数
            part_a = max_freq_gram_2.split(" ")[0]
            repeat = 0
            for word in corpus:
                repeat += self.apperInStr(part_a, word) * corpus[word]
                # if repeat:
                #     break
            # 合并前的a在语料库中不再单独出现，则将其从result中去除
            if repeat == 0:
                bpe_result.pop(part_a)

            # 如果corpus中还存在单独的a,也就是说，result中也还有a,则需要更新result中a的个数
            else:
                bpe_result[part_a] = repeat

            part_b = max_freq_gram_2.split(" ")[1]
            # print("{}:{}--{}".format(max_freq_gram_2, part_a, part_b))

            repeat = 0
            for word in corpus:
                repeat += self.apperInStr(part_b, word) * corpus[word]
            if repeat == 0:
                if part_b in bpe_result:
                    bpe_result.pop(part_b)
                else:
                    print("not found in result:{}".format(part_b))
            else:
                if part_b == "oph" or len(bpe_result) >= 3189:
                    aaaa = 1
                bpe_result[part_b] = repeat

            print("字典大小:{},合并 {}:{}".format(len(bpe_result), max_freq_gram_2, max_freq))
            # print(corpus)
            # print(gram_2)
            # print(bpe_result)

            if len(bpe_result) % 1000 == 0:
                saved_result = sorted(bpe_result.items(), key=lambda d: d[1], reverse=True)
                self.saveToFile(saved_result, output_path)

        return bpe_result

    def saveToFile(self, result, output_path):
        path = "../data/{}_{}.txt".format(output_path, len(result))
        with open(path, 'w', encoding='utf-8') as f:
            for item in result:
                tmp1 = item[0]
                # if char_end in item[0]:
                #     tmp1=item[0].replace(char_end,"")
                # else:
                #     tmp1=item[0]+bpe_symbol
                f.write("{}:{}\n".format(tmp1, item[1]))
        print("保存:{}".format(path))


if __name__ == '__main__':
    corpus_eng_path = "../data/en.txt"
    corpus_de_path = "../data/de.txt"

    bpe = BPE(size=6000) # bpe词典大小设置
    eng = bpe.loadData(corpus_eng_path)
    result = bpe.split(eng, "en")

    bpe = BPE(size=6000)
    de = bpe.loadData(corpus_de_path)
    result = bpe.split(de, "de")
