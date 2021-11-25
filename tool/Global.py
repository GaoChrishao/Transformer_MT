# *_*coding:utf-8 *_*
default_batch_size = 10  # 训练批次大小。
epochs = 20  # 训练迭代轮次。

# 是否使用GPU
use_gpu = True

gpu_index = 0  # 创建摸摸胸训练时，使用的GPU序号
map_gpu_index = 0  # 载入模型时，使用的GPU序号（如果设备没变，则与gpu_index一致）

# encoder和decoder的层数
num_layers = 3

# 多头注意力中的头数
num_heads = 4

# 字嵌入和位置嵌入的维度
d_model = 256
embedding_dim = d_model

# 全连接
d_ff = d_model * 4

# 特殊字符
char_space = ' '
char_start = '<start>'
char_end = '<end>'
char_unknown = '<?>'
word_end = '<e>'  # bpe标志

# 数据集
corpus_encoder_path = './data/de.txt'
corpus_decoder_path = './data/en.txt'

# bpe切分后的数据集
combined_vocab_path = './data/2k/de2en_2k_vocab.txt'

# bpe字典
encoder_bpe_dic_path = './data/2k/de_2000.txt'
decoder_bpe_dic_path = './data/2k/en_2000.txt'

# 重新切分并预处理好的数据集
train_file_path = "./data/2k/train_2k.txt"
valid_file_path = "./data/2k/valid_2k.txt"
test_file_path = "./data/2k/test_2k.txt"

# 训练集字典信息
data_path_vocab_desc = './data/2k/corps_2k_desc.txt'

# 保存模型名称
modelName = "de2en_2k"


# 模型存储路径
def modelPath(epoch):
    return './save/' + modelName + '_%04d' % (epoch + 1) + '.pt'


# 打印训练进度
def printProgress(epoch, prog, batch_no, batch_all, batch_size, loss, accu, lr):
    print('\rEpoch:%04d  prog:%.4f%% batch:%d/%d batch_size:%d mean_loss=%.6f mean_accu=%.2f%% lr=%.6f' % (
        epoch + 1, prog, batch_no, batch_all, batch_size, loss, accu, lr), end="")


# 输出参数到文件
def writeParametersToFile(n_layers, n_heads, d_model, d_ff, batch_size, encoder_len, decoder_len):
    progress = 'n_layers' + '%d' % (n_layers) + \
               '  n_heads:%d' % (n_heads) + \
               '  d_model:%d' % (d_model) + \
               '  d_ff:%d' % (d_ff) + \
               '  batch_size:%d' % (batch_size) + \
               '  encoder_len:%d' % (encoder_len) + \
               '  decoder_len:%d' % (decoder_len) + "\n"
    with open('./save/' + modelName + '.txt', 'a') as f:
        f.write('\n')
        f.write(progress)


# 输出训练进度到文件
def writeProgreeToFile(epoch, batch_all, loss, train_accu, valid_accu, lr):
    progress = 'Epoch:' + '%04d' % (epoch + 1) + \
               '  batch:%d' % (batch_all) + \
               '  loss=' + '{:.6f}'.format(loss) + \
               '  train_accu=' + '{:.6f}'.format(train_accu) + \
               '  valid_accu=' + '{:.6f}'.format(valid_accu) + \
               '  lr=' + '{:.6f}\n'.format(lr)
    with open('./save/' + modelName + '.txt', 'a') as f:
        f.write(progress)
