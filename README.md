# Transformer机器翻译系统

**使用transformer构建的机器翻译系统**

**介绍：**

> 本项目为语言分析与机器翻译作业，
>
> 数据集：iwslt’14 de-en
>
> 训练模型及预处理后训练集下载地址：https://www.aliyundrive.com/s/d9PJMWNQh8Z

**运行环境:**

> pytorch_gpu:1.10
>
> nltk

**目录结构：**

```sh
├── data 
│   ├── 2k #bpe大小为2k的数据集
│   ├── 5k #bpe大小为5k的数据集
│   ├── de.txt #对原始的数据集进行预处理
│   ├── en.txt
│   └── origin_data_set #原始的数据集
├── model
│   └── Transformer.py #Transformer模型文件
├── save #训练结果过程和训练结果保存目录
├── tool
│   ├── DataTool.py #数据载入工具类
│   ├── Global.py #Transformer模型参数及 数据集目录配置
│   └── TrainTool.py # Warm-up函数
├── 数据处理  #以下三个文件，按顺序执行，预处理数据
│   ├── S0_修改原始数据格式.py
│   ├── S1_子词切分.py
│   └── S2_数据集生成.py
├── 测试.py #模型参数
└── 训练.py #模型训练
```

**实现部分：**

>使用pytorch框架搭建Transformer模型（Pre-Norm结构）
>
>实现BPE子词切分算法
>
>实现Beam Search算法
>
>对模型参数（超参数）进行设置、Warm-up、学习率初始化等
>
>对实验结果计算BLEU得分

## 1、训练

### 1.1 从头训练

1. 配置模型参数

   ```sh
   vim Globay.py
   ```

2. 修改train.py

   ```python
   init_model_number=0 #表示从头开始训练
   ```

3. 开始训练

   ```sh
   python 训练.py
   ```

### 1.2 载入保存点，继续训练

1. 配置模型参数，确保与训练时的模型参数一致

   ```sh
   vim Globay.py
   ```

2. 修改train.py

   ![image-20211123135547435](https://i.loli.net/2021/11/23/CbzYcq1gQXyT6o7.png)

   ```python
   init_model_number=10 #表示载入epoch=10的模型文件，继续训练
   model_path = './save/de2en_5k_%04d.pt' % init_model_number #de2en_5k为模型名
   step = 2182 * init_model_number # 2182为一轮的batch数量
   ```

3. 开始执行

   ```sh
   python 训练.py
   ```

### 1.3 训练过程

此处训练参数设置为3层，4头，256维度，bpe大小为2k

```sh
gpu模式
载入已有字典
encoder_chars: 2004
decoder_chars: 2004
max_enc_seq_length: 363
max_dec_seq_length: 278
数据集大小:172530
数据集大小:872
batch num:1120
Epoch:0001  prog:100.0017% batch:1120/1120 batch_size:180 mean_loss=5.996997 mean_accu=11.41% lr=0.000098
进行测试中...batch num:44
valid accuracy:22.38 % test loss:4.4169
......
......
batch num:1120
Epoch:0020  prog:100.0017% batch:1120/1120 batch_size:200 mean_loss=2.138918 mean_accu=68.88% lr=0.000418
进行测试中...batch num:44
valid accuracy:70.41 % test loss:1.4048
```



### 1.4 查看训练过程

```sh
cat save/de2en_2k.txt

n_layers3  n_heads:4  d_model:256  d_ff:1024  batch_size:20  encoder_len:363  decoder_len:278
Epoch:0001  batch:1120  loss=5.996997  train_accu=11.412931  valid_accu=22.383728  lr=0.000098
Epoch:0002  batch:1120  loss=4.755715  train_accu=24.517178  valid_accu=36.618428  lr=0.000196
Epoch:0003  batch:1120  loss=3.974497  train_accu=35.328851  valid_accu=45.392770  lr=0.000293
Epoch:0004  batch:1120  loss=3.470239  train_accu=43.350113  valid_accu=52.148458  lr=0.000391
Epoch:0005  batch:1120  loss=3.135638  train_accu=49.315405  valid_accu=56.918749  lr=0.000489
Epoch:0006  batch:1120  loss=2.912801  train_accu=53.593547  valid_accu=59.448717  lr=0.000587
Epoch:0007  batch:1120  loss=2.763752  train_accu=56.515822  valid_accu=62.279953  lr=0.000685
Epoch:0008  batch:1120  loss=2.642074  train_accu=58.967584  valid_accu=63.859193  lr=0.000660
Epoch:0009  batch:1120  loss=2.534271  train_accu=61.111251  valid_accu=65.178845  lr=0.000623
Epoch:0010  batch:1120  loss=2.453718  train_accu=62.700495  valid_accu=66.578826  lr=0.000591
Epoch:0011  batch:1120  loss=2.393994  train_accu=63.887431  valid_accu=67.546647  lr=0.000563
Epoch:0012  batch:1120  loss=2.344394  train_accu=64.858695  valid_accu=67.881313  lr=0.000539
Epoch:0013  batch:1120  loss=2.303971  train_accu=65.646550  valid_accu=68.559155  lr=0.000518
Epoch:0014  batch:1120  loss=2.270668  train_accu=66.321382  valid_accu=68.881203  lr=0.000499
Epoch:0015  batch:1120  loss=2.240943  train_accu=66.875999  valid_accu=69.428752  lr=0.000482
Epoch:0016  batch:1120  loss=2.216345  train_accu=67.376881  valid_accu=69.645483  lr=0.000467
Epoch:0017  batch:1120  loss=2.193870  train_accu=67.818207  valid_accu=69.951239  lr=0.000453
Epoch:0018  batch:1120  loss=2.173374  train_accu=68.200069  valid_accu=70.102368  lr=0.000440
Epoch:0019  batch:1120  loss=2.155479  train_accu=68.574355  valid_accu=70.356942  lr=0.000428
Epoch:0020  batch:1120  loss=2.138918  train_accu=68.877916  valid_accu=70.409328  lr=0.000418
```

## 2、测试BLEU得分

1. 配置模型参数，确保与训练时的模型参数一致

   ```sh
   vim Globay.py
   ```

2. 修改测试.py

   ```python
   model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
   m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
   model.load_state_dict(m_state_dict)
   k = 3
   ```

   de2en_2k_0020.pt为模型文件

   map_gpu_index设置使用的gpu序号

   k为书搜索宽度

3. 运行测试

   ```sh
   python 测试.py
   
   原文：nun , das bringt ihnen keinen doktortitel in harvard , aber es ist viel interessanter als stamina zu zählen . 
   参考译文：now , that &apos;s not going to get you a ph.d. at harvard , but it &apos;s a lot more interesting than counting stamens . 
   
   -12.349,now , it doesn &apos;t get you a doctor in harvard , but it &apos;s much more interesting to count than stamina . 
   -12.397,now , this doesn &apos;t get you a doctor in harvard , but it &apos;s much more interesting to count than stamina . 
   -13.909,now , it doesn &apos;t get you a doctor in harvard , but it &apos;s much more interesting to count as a stamina . 
   
   ......
   ......
   
   原文：es war auf jeden fall eine herausforderung . 
   参考译文：so , certainly the challenge was there . 
   
   -4.524,it was a challenge anyway . 
   -2.915,it was certainly a challenge . 
   -3.620,it was definitely a challenge . 
    
   原文：wir haben bis jetzt wirklich nicht die richtigen worte , sie zu beschreiben . 
   参考译文：we don &apos;t have words , really , to describe it yet . 
   
   -4.718,we really don &apos;t have the right words to describe them . 
   -5.074,we really don &apos;t have the right words to describe it . 
   -5.088,we don &apos;t really have the right words to describe them . 
    
   原文：ich spreche oft darüber , situationen zu &quot; entdenken &quot; . 
   参考译文：i talk about unthinking situations all the time . 
   
   -2.874,i often talk about finding situations . 
   -2.882,i often talk about discovering situations . 
   -4.159,i &apos;m often talking about discovering situations . 
    
   bleu：0.5828,0.3548,0.2298,0.1537
   mean bleu：0.3303
   ```
   

