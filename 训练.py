# *_*coding:utf-8 *_*
import math
import sys
import torch.nn as nn

from tool.TrainTool import adjust_learning_rate
from model.Transformer import Transformer
from tool.DataTool import *


# 每轮训练结束时，进行测试
def test_model(model, test_data):
    model.eval()

    test_all_loss = 0
    test_all_accuracy = 0
    test_batch_no = 0

    with torch.no_grad():
        for batch_size, enc_inputs, dec_inputs, dec_outputs in data_generator(
                source_vocab2id,
                target_vocab2id,
                test_data,
                default_batch_size,
                max_enc_seq_length,
                max_dec_seq_length,
                False,
        ):
            if enc_inputs is None:
                break
            test_batch_no += 1
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)

            _,outs = model(enc_inputs, dec_inputs)
            dec_outputs_reshaped = dec_outputs.reshape(-1)

            # 获取预测正确的个数

            test_all_accuracy += calculate_accuracy(outs, dec_outputs_reshaped)

            loss = criterion(outs, dec_outputs_reshaped)
            test_all_loss += loss

    return test_all_accuracy / test_batch_no, test_all_loss / test_batch_no


# 计算准确率，需要去除padding=0
def calculate_accuracy(model_predict, target, ignore_index=0):
    # target中 非0的地方填充为1
    non_pad_mask = target.ne(ignore_index)
    # 得到target中1的数量，即有效的词的数量
    word_num = non_pad_mask.sum().item()
    # 得到预测正确的数量
    predict_correct_num = model_predict.max(dim=-1).indices.eq(target).masked_select(non_pad_mask).sum().item()

    return predict_correct_num / word_num * 100


if __name__ == '__main__':
    if use_gpu:
        device = torch.device("cuda:{}".format(gpu_index))
        print("gpu模式")
    else:
        device = torch.device("cpu")
        print("cpu模式")

    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    print('encoder_chars:', len(encoder_chars))
    print('decoder_chars:', len(decoder_chars))
    print('max_enc_seq_length:', max_enc_seq_length)
    print('max_dec_seq_length:', max_dec_seq_length)

    source_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    source_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    target_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    target_id2vocab = {i: word for i, word in enumerate(decoder_chars)}

    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(params=model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    init_model_number = 6
    if init_model_number > 0:
        model_path = './save/de2en_2k_%04d.pt' % init_model_number
        m_state_dict = torch.load(model_path, map_location="cuda:{}".format(map_gpu_index))
        model.load_state_dict(m_state_dict)
        step = 2182 * init_model_number  # 此处需要手动计算！，开启auto fit batch_size后，需要手动修改参数
    else:
        step = 0

    # 把网络、损失函数转换到GPU上
    model = model.to(device)
    criterion = criterion.to(device)

    writeParametersToFile(num_layers,
                          num_heads,
                          d_model,
                          d_ff, default_batch_size, max_enc_seq_length, max_dec_seq_length)

    train_samples_num, train_data = load_dataset(train_file_path)
    valid_samples_num, valid_data = load_dataset(valid_file_path)


    for now_epoch in range(init_model_number, epochs):
        batch_no = 0  # 当前训练的batch编号(在开启auto fit batch size后，该参数无参考意义)
        all_loss = 0  # 用于计算mean_loss
        all_accuracy = 0

        has_trained_samples = 0  # 训练了的行数,由于计算progress

        model.train()

        for batch_num, enc_inputs, dec_inputs, dec_outputs in data_generator(
                source_vocab2id,
                target_vocab2id,
                train_data,
                default_batch_size,
                max_enc_seq_length,
                max_dec_seq_length,
                True,
        ):
            if enc_inputs is None:
                break
            step += 1
            batch_no += 1
            current_batch_size = len(enc_inputs)

            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            dec_outputs = dec_outputs.to(device)

            # 清除上一步的梯度
            optimizer.zero_grad()

            # 训练，并得到输出和loss
            _,outs = model(enc_inputs, dec_inputs)

            dec_outputs_reshaped = dec_outputs.reshape(-1)
            loss = criterion(outs, dec_outputs_reshaped)

            # 获取预测正确的个数
            all_accuracy += calculate_accuracy(outs, dec_outputs_reshaped)
            all_loss += loss

            adjust_learning_rate(d_model, step, optimizer)  # 调整学习率

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            # test
            has_trained_samples += current_batch_size

            progress = printProgress(now_epoch,
                                     has_trained_samples * 100 / train_samples_num,
                                     batch_no,
                                     batch_num,
                                     current_batch_size,
                                     all_loss / batch_no,
                                     all_accuracy / batch_no,
                                     lr)

            loss.backward()
            optimizer.step()  # 调整学习率

            # torch.cuda.empty_cache()  # 清理cuda缓存，尝试解决 out of memory这个问题

        # 设置loss小于3时才保存模型
        #if (now_epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), modelPath(now_epoch))
        print("\n进行测试中...", end='')
        valid_accuracy, valid_loss = test_model(model, valid_data)
        print("\rvalid accuracy:%.2f %% test loss:%.4f" % (valid_accuracy, valid_loss))

        writeProgreeToFile(now_epoch, batch_num, all_loss / batch_no, all_accuracy / batch_no, valid_accuracy, lr)
        print("")
