# 一、基于PaddleNLP预训练ERNIE模型优化中文地址要素解析

[“英特尔创新大师杯”深度学习挑战赛 赛道2：CCKS2021中文NLP地址要素解析-天池大赛-阿里云天池](https://tianchi.aliyun.com/competition/entrance/531900/information)


## 1.赛题描述
中文地址要素解析任务的目标即将一条地址分解为上述几个部分的详细标签，如：

输入：浙江省杭州市余杭区五常街道文一西路969号淘宝城5号楼，放前台
输出：Province=浙江省 city=杭州市 district=余杭区 town=五常街道 road=文一西路road_number=969号 poi=淘宝城 house_number=5号楼 other=，放前台

## 2.数据说明
标注数据集由训练集、验证集和测试集组成，整体标注数据大约2万条左右。地址数据通过抓取公开的地址信息（如黄页网站等）获得， 均通过众包标注生成，详细标注规范将会在数据发布时一并给出。


## 3.命名实体识别介绍
命名实体识别是NLP中一项非常基础的任务，是信息提取、问答系统、句法分析、机器翻译等众多NLP任务的重要基础工具。命名实体识别的准确度，决定了下游任务的效果，是NLP中的一个基础问题。在NER任务提供了两种解决方案，一类LSTM/GRU + CRF，RNN类的模型来抽取底层文本的信息，而CRF(条件随机场)模型来学习底层Token之间的联系；另外一类是通过预训练模型，例如ERNIE，BERT模型，直接来预测Token的标签信息。

本项目将演示，如何使用PaddleNLP语义预训练模型ERNIE完成从快递单中抽取姓名、电话、省、市、区、详细地址等内容，形成结构化信息。辅助物流行业从业者进行有效信息的提取，从而降低客户填单的成本，完成比赛。

# 二、RNN命名实体识别概念
在2017年之前，工业界和学术界对NLP文本处理依赖于序列模型[Recurrent Neural Network (RNN)](https://baike.baidu.com/item/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/23199490?fromtitle=RNN&fromid=5707183&fr=aladdin).

<p align="center">
<img src="http://colah.github.io/posts/2015-09-NN-Types-FP/img/RNN-general.png" width="40%" height="30%"> <br />
</p><br><center>图1：RNN示意图</center></br>

[基于BiGRU+CRF的快递单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/1317771)项目介绍了如何使用序列模型完成快递单信息抽取任务。
<br>

近年来随着深度学习的发展，模型参数的数量飞速增长。为了训练这些参数，需要更大的数据集来避免过拟合。然而，对于大部分NLP任务来说，构建大规模的标注数据集非常困难（成本过高），特别是对于句法和语义相关的任务。相比之下，大规模的未标注语料库的构建则相对容易。为了利用这些数据，我们可以先从其中学习到一个好的表示，再将这些表示应用到其他任务中。最近的研究表明，基于大规模未标注语料库的预训练模型（Pretrained Models, PTM) 在NLP任务上取得了很好的表现。

近年来，大量的研究表明基于大型语料库的预训练模型（Pretrained Models, PTM）可以学习通用的语言表示，有利于下游NLP任务，同时能够避免从零开始训练模型。随着计算能力的发展，深度模型的出现（即 Transformer）和训练技巧的增强使得 PTM 不断发展，由浅变深。


<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/327f44ff3ed24493adca5ddc4dc24bf61eebe67c84a6492f872406f464fde91e" width="60%" height="50%"> <br />
</p><br><center>图2：预训练模型一览，图片来源于：https://github.com/thunlp/PLMpapers</center></br>
                                                                                                                             
本示例展示了以ERNIE([Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223))代表的预训练模型如何Finetune完成序列标注任务。

# 三、数据分析

## 1.PaddleNLP环境准备


```python
!pip install --upgrade paddlenlp
```


```python
from functools import partial

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from utils import convert_example, evaluate, predict, load_dict
```

## 2.数据整理


```python
!unzip 'data/data94613/“英特尔创新大师杯”深度学习挑战赛 赛道2：CCKS2021中文NLP地址要素解析.zip'
```


```python
!mv 'б░╙в╠╪╢√┤┤╨┬┤є╩ж▒нб▒╔ю╢╚╤з╧░╠Ї╒╜╚№ ╚№╡└2г║CCKS2021╓╨╬─NLP╡╪╓╖╥к╦╪╜т╬Ў' dataset
!mv 'dataset/╓╨╬─╡╪╓╖╥к╦╪╜т╬Ў▒ъ╫в╣ц╖╢.pdf' dastaset/中文地址要素解析标注规范.pdf
```

## 3.数据查看


```python
!head -n10 dataset/train.conll
```

    浙 B-prov
    江 E-prov
    杭 B-city
    州 I-city
    市 E-city
    江 B-district
    干 I-district
    区 E-district
    九 B-town
    堡 I-town



```python
!head -n10  dataset/dev.conll
```

    杭 B-city
    州 E-city
    五 B-poi
    洲 I-poi
    国 I-poi
    际 E-poi
    
    浙 B-prov
    江 I-prov
    省 E-prov



```python
!head dataset/final_test.txt
```

    1朝阳区小关北里000-0号
    2朝阳区惠新东街00号
    3朝阳区南磨房路与西大望路交口东南角
    4朝阳区潘家园南里00号
    5朝阳区向军南里二巷0号附近
    6朝阳区多处营业网点
    7朝阳区多处营业网点
    8朝阳区多处营业网点
    9朝阳区北三环中路00号商房大厦0楼
    10朝阳区孙河乡康营家园00区北侧底商


## 4.数据格式调整


```python
import os

def format_data(source_filename, target_filename):
    datalist=[]
    with open(source_filename, 'r', encoding='utf-8') as f:
        lines=f.readlines()
    words=''
    labels=''
    flag=0
    for line in lines:  
        if line == '\n':
            item=words+'\t'+labels+'\n'
            # print(item)
            datalist.append(item)
            words=''
            labels=''
            flag=0
            continue
        word, label = line.strip('\n').split(' ')
        if flag==1:
            words=words+'\002'+word
            labels=labels+'\002'+label
        else:
            words=words+word
            labels=labels+label
            flag=1
    with open(target_filename, 'w', encoding='utf-8') as f:
        lines=f.writelines(datalist)
    print(f'{source_filename}文件格式转换完毕，保存为{target_filename}')
```


```python
format_data('dataset/dev.conll', 'dataset/dev.txt')
format_data(r'dataset/train.conll', r'dataset/train.txt')
```

    dataset/dev.conll文件格式转换完毕，保存为dataset/dev.txt
    dataset/train.conll文件格式转换完毕，保存为dataset/train.txt



```python
!head dataset/dev.txt
```

    杭州五洲国际	B-cityE-cityB-poiI-poiI-poiE-poi
    浙江省杭州市余杭乔司街道博卡路0号博卡制衣	B-provI-provE-provB-cityI-cityE-cityB-districtE-districtB-townI-townI-townE-townB-roadI-roadE-roadB-roadnoE-roadnoB-poiI-poiI-poiE-poi
    浙江诸暨市暨阳八一新村00幢	B-provE-provB-districtI-districtE-districtB-townE-townB-poiI-poiI-poiE-poiB-housenoI-housenoE-houseno
    杭州市武林广场杭州大厦商城A座九层	B-cityI-cityE-cityB-poiI-poiI-poiE-poiB-subpoiI-subpoiI-subpoiE-subpoiB-subpoiE-subpoiB-housenoE-housenoB-floornoE-floorno
    浙江省杭州市拱墅区登云路0000号时代电子市场	B-provI-provE-provB-cityI-cityE-cityB-districtI-districtE-districtB-roadI-roadE-roadB-roadnoI-roadnoI-roadnoI-roadnoE-roadnoB-poiI-poiI-poiI-poiI-poiE-poi
    浙江省宁波市慈溪市宗汉街道联丰公寓00栋	B-provI-provE-provB-cityI-cityE-cityB-districtI-districtE-districtB-townI-townI-townE-townB-poiI-poiI-poiE-poiB-housenoI-housenoE-houseno
    浙江省温州市鹿城区劳务市场跨境电商园00楼艺网科技有限公司	B-provI-provE-provB-cityI-cityE-cityB-districtI-districtE-districtB-poiI-poiI-poiE-poiB-devzoneI-devzoneI-devzoneI-devzoneE-devzoneB-floornoI-floornoE-floornoB-subpoiI-subpoiI-subpoiI-subpoiI-subpoiI-subpoiI-subpoiE-subpoi
    康中路00号康城工业园00幢0楼	B-roadI-roadE-roadB-roadnoI-roadnoE-roadnoB-devzoneI-devzoneI-devzoneI-devzoneE-devzoneB-housenoI-housenoE-housenoB-floornoE-floorno
    金华永康市城西工业区蓝天路坊培电脑	B-cityE-cityB-districtI-districtE-districtB-devzoneI-devzoneI-devzoneI-devzoneE-devzoneB-roadI-roadE-roadB-poiI-poiI-poiE-poi
    宜山人民路0000号后栋纸巾厂	B-townE-townB-roadI-roadE-roadB-roadnoI-roadnoI-roadnoI-roadnoE-roadnoB-housenoE-housenoB-poiI-poiE-poi


## 5.加载自定义数据集

推荐使用MapDataset()自定义数据集。


```python
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                words, labels = line.strip('\n').split('\t')
                words = words.split('\002')
                labels = labels.split('\002')
                yield words, labels

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]        
```


```python
# Create dataset, tokenizer and dataloader.
train_ds, dev_ds = load_dataset(datafiles=(
        './dataset/train.txt', './dataset/dev.txt'))
```


```python
for i in range(5):
    print(train_ds[i])
```

    (['浙', '江', '省', '温', '州', '市', '平', '阳', '县', '海', '西', '镇', '宋', '埠', '公', '园', '南', '路', '0', '0', '0', '0', '号'], ['B-prov', 'I-prov', 'E-prov', 'B-city', 'I-city', 'E-city', 'B-district', 'I-district', 'E-district', 'B-town', 'I-town', 'E-town', 'B-poi', 'I-poi', 'I-poi', 'E-poi', 'B-road', 'E-road', 'B-roadno', 'I-roadno', 'I-roadno', 'I-roadno', 'E-roadno'])
    (['浙', '江', '省', '余', '姚', '市', '模', '具', '城', '金', '型', '路', '0', '0', '0', '号', '_', '样', '样', '红', '0', 'A', '打', '印'], ['B-prov', 'I-prov', 'E-prov', 'B-district', 'I-district', 'E-district', 'B-poi', 'I-poi', 'E-poi', 'B-road', 'I-road', 'E-road', 'B-roadno', 'I-roadno', 'I-roadno', 'E-roadno', 'O', 'B-subpoi', 'I-subpoi', 'I-subpoi', 'I-subpoi', 'I-subpoi', 'I-subpoi', 'E-subpoi'])
    (['浙', '江', '省', '杭', '州', '市', '江', '干', '区', '白', '杨', '街', '道', '下', '沙', '开', '发', '区', '世', '茂', '江', '滨', '花', '园', '峻', '景', '湾', '0', '0', '幢'], ['B-prov', 'I-prov', 'E-prov', 'B-city', 'I-city', 'E-city', 'B-district', 'I-district', 'E-district', 'B-town', 'I-town', 'I-town', 'E-town', 'B-devzone', 'I-devzone', 'I-devzone', 'I-devzone', 'E-devzone', 'B-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'E-poi', 'B-subpoi', 'I-subpoi', 'E-subpoi', 'B-houseno', 'I-houseno', 'E-houseno'])
    (['秋', '菱', '路', '浙', '江', '兰', '溪', '金', '立', '达', '框', '业', '有', '限', '公', '司'], ['B-road', 'I-road', 'E-road', 'B-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'E-poi'])
    (['南', '湖', '区', '中', '环', '南', '路', '和', '花', '园', '路', '交', '叉', '口', '嘉', '兴', '市', '城', '乡', '规', '划', '建', '设', '管', '理', '委', '员', '会'], ['B-district', 'I-district', 'E-district', 'B-road', 'I-road', 'I-road', 'E-road', 'O', 'B-road', 'I-road', 'E-road', 'B-intersection', 'I-intersection', 'E-intersection', 'B-city', 'I-city', 'E-city', 'B-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'I-poi', 'E-poi'])


## 6 label标签表构建
每条数据包含一句文本和这个文本中每个汉字以及数字对应的label标签，具体对应关系见 **中文地址要素解析标注规范.pdf**

之后，还需要对输入句子进行数据处理，如切词，映射词表id等。


```python
def gernate_dic(source_filename1, source_filename2, target_filename):
    data_list=[]

    with open(source_filename1, 'r', encoding='utf-8') as f:
        lines=f.readlines()

    for line in lines:
        if line != '\n':
            dic=line.strip('\n').split(' ')[-1]
            if dic+'\n' not in data_list:
                data_list.append(dic+'\n')
    
    with open(source_filename2, 'r', encoding='utf-8') as f:
        lines=f.readlines()

    for line in lines:
        if line != '\n':
            dic=line.strip('\n').split(' ')[-1]
            if dic+'\n' not in data_list:
                data_list.append(dic+'\n')

    with open(target_filename, 'w', encoding='utf-8') as f:
        lines=f.writelines(data_list)    
```


```python
# 从dev文件生成dic
gernate_dic('dataset/train.conll', 'dataset/dev.conll', 'dataset/mytag.dic')
# gernate_dic('dataset/dev.conll', 'dataset/mytag_dev.dic')
```


```python
# 查看生成的dic文件
!cat dataset/mytag.dic
```

    B-prov
    E-prov
    B-city
    I-city
    E-city
    B-district
    I-district
    E-district
    B-town
    I-town
    E-town
    B-community
    I-community
    E-community
    B-poi
    E-poi
    I-prov
    I-poi
    B-road
    E-road
    B-roadno
    I-roadno
    E-roadno
    I-road
    O
    B-subpoi
    I-subpoi
    E-subpoi
    B-devzone
    I-devzone
    E-devzone
    B-houseno
    I-houseno
    E-houseno
    B-intersection
    I-intersection
    E-intersection
    B-assist
    I-assist
    E-assist
    B-cellno
    I-cellno
    E-cellno
    B-floorno
    E-floorno
    S-assist
    I-floorno
    B-distance
    I-distance
    E-distance
    B-village_group
    E-village_group
    I-village_group
    S-poi
    S-intersection
    S-district
    S-community


## 7.数据处理

预训练模型ERNIE对中文数据的处理是以字为单位。PaddleNLP对于各种预训练模型已经内置了相应的tokenizer。指定想要使用的模型名字即可加载对应的tokenizer。

tokenizer作用为将原始输入文本转化成模型model可以接受的输入数据形式。


<p align="center">
<img src="https://bj.bcebos.com/paddlehub/paddlehub-img/ernie_network_1.png" hspace='10'/> <br />
</p>


<p align="center">
<img src="https://bj.bcebos.com/paddlehub/paddlehub-img/ernie_network_2.png" hspace='10'/> <br />
</p>
<br><center>图3：ERNIE模型示意图</center></br>


```python
label_vocab = load_dict('./dataset/mytag.dic')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)

train_ds.map(trans_func)
dev_ds.map(trans_func)
print (train_ds[0])
```

    [2021-06-28 13:26:34,755] [    INFO] - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt
    100%|██████████| 90/90 [00:00<00:00, 4654.25it/s]


    ([1, 1382, 409, 244, 565, 404, 99, 157, 507, 308, 233, 213, 484, 945, 3074, 53, 509, 219, 216, 540, 540, 540, 540, 500, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 25, [24, 0, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 17, 17, 15, 18, 19, 20, 21, 21, 21, 22, 24])


### 数据读入

使用`paddle.io.DataLoader`接口多线程异步加载数据。


```python
ignore_label = -1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)

train_loader = paddle.io.DataLoader(
    dataset=train_ds,
    batch_size=300,
    return_list=True,
    collate_fn=batchify_fn)
dev_loader = paddle.io.DataLoader(
    dataset=dev_ds,
    batch_size=300,
    return_list=True,
    collate_fn=batchify_fn)
```

# 四、PaddleNLP一键加载预训练模型

## 1.加载预训练模型

快递单信息抽取本质是一个序列标注任务，PaddleNLP对于各种预训练模型已经内置了对于下游任务文本分类Fine-tune网络。以下教程以ERNIE为预训练模型完成序列标注任务。

`paddlenlp.transformers.ErnieForTokenClassification()`一行代码即可加载预训练模型ERNIE用于序列标注任务的fine-tune网络。其在ERNIE模型后拼接上一个全连接网络进行分类。

`paddlenlp.transformers.ErnieForTokenClassification.from_pretrained()`方法只需指定想要使用的模型名称和文本分类的类别数即可完成定义模型网络。


```python
# Define the model netword and its loss
model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))
```

    [2021-06-28 13:26:34,864] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-1.0
    [2021-06-28 13:26:34,866] [    INFO] - Downloading ernie_v1_chn_base.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams
    100%|██████████| 392507/392507 [00:08<00:00, 48559.94it/s]
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


PaddleNLP不仅支持ERNIE预训练模型，还支持BERT、RoBERTa、Electra等预训练模型。
下表汇总了目前PaddleNLP支持的各类预训练模型。您可以使用PaddleNLP提供的模型，完成文本分类、序列标注、问答等任务。同时我们提供了众多预训练模型的参数权重供用户使用，其中包含了二十多种中文语言模型的预训练权重。中文的预训练模型有`bert-base-chinese, bert-wwm-chinese, bert-wwm-ext-chinese, ernie-1.0, ernie-tiny, gpt2-base-cn, roberta-wwm-ext, roberta-wwm-ext-large, rbt3, rbtl3, chinese-electra-base, chinese-electra-small, chinese-xlnet-base, chinese-xlnet-mid, chinese-xlnet-large, unified_transformer-12L-cn, unified_transformer-12L-cn-luge`等。

更多预训练模型参考：[PaddleNLP Transformer API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/transformers.md)。

更多预训练模型fine-tune下游任务使用方法，请参考：[examples](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples)。

## 2.设置Fine-Tune优化策略，模型配置
适用于ERNIE/BERT这类Transformer模型的迁移优化学习率策略为warmup的动态学习率。

<p align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/2bc624280a614a80b5449773192be460f195b13af89e4e5cbaf62bf6ac16de2c" width="40%" height="30%"/> <br />
</p><br><center>图4：动态学习率示意图</center></br>




```python
metric = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
loss_fn = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
```

# 五、模型训练与评估

## 1.训练模型

模型训练的过程通常有以下步骤：

1. 从dataloader中取出一个batch data
2. 将batch data喂给model，做前向计算
3. 将前向计算结果传给损失函数，计算loss。将前向计算结果传给评价方法，计算评价指标。
4. loss反向回传，更新梯度。重复以上步骤。

每训练一个epoch时，程序将会评估一次，评估当前模型训练的效果。


```python
step = 0
for epoch in range(50):
    for idx, (input_ids, token_type_ids, length, labels) in enumerate(train_loader):
        logits = model(input_ids, token_type_ids)
        loss = paddle.mean(loss_fn(logits, labels))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        step += 1
        print("epoch:%d - step:%d - loss: %f" % (epoch, step, loss))
    evaluate(model, metric, dev_loader)

    paddle.save(model.state_dict(),
                './checkpoint/model_%d.pdparams' % step)
```

```
epoch:49 - step:1832 - loss: 0.057792
epoch:49 - step:1833 - loss: 0.053191
epoch:49 - step:1834 - loss: 0.051053
epoch:49 - step:1835 - loss: 0.054221
epoch:49 - step:1836 - loss: 0.036712
epoch:49 - step:1837 - loss: 0.038394
epoch:49 - step:1838 - loss: 0.045484
epoch:49 - step:1839 - loss: 0.068006
epoch:49 - step:1840 - loss: 0.039057
epoch:49 - step:1841 - loss: 0.049253
epoch:49 - step:1842 - loss: 0.049330
epoch:49 - step:1843 - loss: 0.051696
epoch:49 - step:1844 - loss: 0.042183
epoch:49 - step:1845 - loss: 0.041376
epoch:49 - step:1846 - loss: 0.040038
epoch:49 - step:1847 - loss: 0.046694
epoch:49 - step:1848 - loss: 0.043038
epoch:49 - step:1849 - loss: 0.046348
epoch:49 - step:1850 - loss: 0.007658
eval precision: 0.997797 - recall: 0.998420 - f1: 0.998109
```

## 2.模型保存


```python
!mkdir ernie_result
model.save_pretrained('./ernie_result')
tokenizer.save_pretrained('./ernie_result')
```

# 六、预测

训练保存好的训练，即可用于预测。如以下示例代码自定义预测数据，调用`predict()`函数即可一键预测。


```python
import numpy as np
import paddle
from paddle.io import DataLoader
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import ErnieTokenizer, ErnieForTokenClassification
from paddlenlp.metrics import ChunkEvaluator
from utils import convert_example, evaluate, predict, load_dict
from functools import partial
```


```python
!head -n20 dataset/final_test.txt
```

    1朝阳区小关北里000-0号
    2朝阳区惠新东街00号
    3朝阳区南磨房路与西大望路交口东南角
    4朝阳区潘家园南里00号
    5朝阳区向军南里二巷0号附近
    6朝阳区多处营业网点
    7朝阳区多处营业网点
    8朝阳区多处营业网点
    9朝阳区北三环中路00号商房大厦0楼
    10朝阳区孙河乡康营家园00区北侧底商
    11朝阳区将台乡雍家村
    12朝阳区安家楼村路
    13朝阳区郎辛庄北路
    14朝阳区酒仙桥路0号院0号楼一层
    15朝阳区十里堡北里南区0号楼0楼
    16朝阳区双桥医院
    17朝阳区五里桥一街甲0号中弘北京像素北区0号楼0单元0000号
    18朝阳区傲城融富中心A座0000
    19朝阳区西坝河西里00号英特公寓A0座0000室
    20朝阳区姚家园路00号院


## 1.定义test数据集


```python
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            # next(fp)  # 没有header，不用Skip header
            for line in fp.readlines():
                ids, words = line.strip('\n').split('\001')
                words=[ch for ch in words]
                # 要预测的数据集没有label，伪造个O，不知道可以不 ，应该后面预测不会用label
                labels=['O' for x in range(0,len(words))]

                yield words, labels
                # yield words

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]      
```


```python
# Create dataset, tokenizer and dataloader.
test_ds = load_dataset(datafiles=('./dataset/final_test.txt'))
```


```python
for i in range(20):
    print(test_ds[i])
```

    (['朝', '阳', '区', '小', '关', '北', '里', '0', '0', '0', '-', '0', '号'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    (['朝', '阳', '区', '惠', '新', '东', '街', '0', '0', '号'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
    (['朝', '阳', '区', '南', '磨', '房', '路', '与', '西', '大', '望', '路', '交', '口', '东', '南', '角'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])


## 2.加载训练好的模型


```python
label_vocab = load_dict('./dataset/mytag.dic')
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab)
test_ds.map(trans_func)
print (test_ds[0])
```


```python
ignore_label = 1
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(),  # seq_len
    Pad(axis=0, pad_val=ignore_label)  # labels
): fn(samples)
```


```python
test_loader = paddle.io.DataLoader(
    dataset=test_ds,
    batch_size=30,
    return_list=True,
    collate_fn=batchify_fn)
```


```python
def my_predict(model, data_loader, ds, label_vocab):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        logits = model(input_ids, seg_ids)
        # print(len(logits[0]))
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(lens.numpy())
    preds ,tags= parse_decodes(ds, pred_list, len_list, label_vocab)
    return preds, tags
```


```python
# Define the model netword and its loss
model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=len(label_vocab))
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))



```python
model_dict = paddle.load('ernie_result/model_state.pdparams')
model.set_dict(model_dict)
```

## 3.预测并保存


```python
from utils import *
preds, tags = my_predict(model, test_loader, test_ds, label_vocab)
```


```python
file_path = "ernie_results.txt"
with open(file_path, "w", encoding="utf8") as fout:
    fout.write("\n".join(preds))
# Print some examples
print(
    "The results have been saved in the file: %s, some examples are shown below: "
    % file_path)
```

    The results have been saved in the file: ernie_results.txt, some examples are shown below: 



```python
print("\n".join(preds[:20]))
```

     B-district I-district E-district B-road I-road I-road E-road B-roadno I-roadno I-roadno I-roadno I-roadno E-roadno
     B-district I-district E-district B-road I-road I-road E-road B-roadno I-roadno E-roadno
     B-district I-district E-district B-road I-road I-road E-road O B-road I-road I-road E-road B-intersection E-intersection B-assist I-assist E-assist
     B-district I-district E-district B-poi I-poi E-poi B-road E-road B-houseno I-houseno E-houseno
     B-district I-district E-district B-road I-road I-road E-road B-road E-road B-roadno E-roadno B-assist E-assist
     B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi
     B-district I-district E-district B-poi I-poi B-poi I-poi I-poi E-poi
     B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi
     B-district I-district E-district B-road I-road I-road I-road E-road B-roadno I-roadno E-roadno B-poi I-poi I-poi E-poi B-floorno E-floorno
     B-district I-district E-district B-town I-town E-town B-poi I-poi I-poi E-poi B-subpoi I-subpoi E-subpoi B-assist E-assist O E-subpoi
     B-district I-district E-district B-town I-town E-town B-community I-community E-community
     B-district I-district E-district B-community I-community I-community E-community O
     B-district I-district E-district B-road I-road I-road I-road E-road
     B-district I-district E-district B-road I-road I-road E-road B-poi I-poi E-poi B-houseno I-houseno E-houseno O O
     B-district I-district E-district B-poi I-poi E-poi I-poi E-poi B-subpoi E-poi B-houseno I-houseno E-houseno B-floorno E-floorno
     B-district I-district E-district B-poi I-poi I-poi E-poi
     B-district I-district E-district B-road I-road I-road I-road E-road B-roadno I-roadno E-roadno B-poi I-poi I-poi I-poi I-poi I-poi I-poi E-poi B-houseno I-houseno E-houseno B-cellno I-cellno E-cellno O I-houseno I-houseno I-houseno E-houseno
     B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi B-houseno E-houseno O O O O
     B-district I-district E-district B-road I-road I-road B-road E-road B-roadno I-roadno E-roadno B-poi I-poi I-poi E-poi B-houseno I-houseno E-houseno O O O O E-floorno
     B-district I-district E-district B-road I-road I-road E-road B-poi I-poi I-poi E-poi



```python
!head ernie_results.txt
```

     B-district I-district E-district B-road I-road I-road E-road B-roadno I-roadno I-roadno I-roadno I-roadno E-roadno
     B-district I-district E-district B-road I-road I-road E-road B-roadno I-roadno E-roadno
     B-district I-district E-district B-road I-road I-road E-road O B-road I-road I-road E-road B-intersection E-intersection B-assist I-assist E-assist
     B-district I-district E-district B-poi I-poi E-poi B-road E-road B-houseno I-houseno E-houseno
     B-district I-district E-district B-road I-road I-road E-road B-road E-road B-roadno E-roadno B-assist E-assist
     B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi
     B-district I-district E-district B-poi I-poi B-poi I-poi I-poi E-poi
     B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi
     B-district I-district E-district B-road I-road I-road I-road E-road B-roadno I-roadno E-roadno B-poi I-poi I-poi E-poi B-floorno E-floorno
     B-district I-district E-district B-town I-town E-town B-poi I-poi I-poi E-poi B-subpoi I-subpoi E-subpoi B-assist E-assist O E-subpoi



```python
!head ./dataset/final_test.txt
```

    1朝阳区小关北里000-0号
    2朝阳区惠新东街00号
    3朝阳区南磨房路与西大望路交口东南角
    4朝阳区潘家园南里00号
    5朝阳区向军南里二巷0号附近
    6朝阳区多处营业网点
    7朝阳区多处营业网点
    8朝阳区多处营业网点
    9朝阳区北三环中路00号商房大厦0楼
    10朝阳区孙河乡康营家园00区北侧底商


## 4.转换保存结果


```python
def main():
    data_list = []
    with open('ernie_results.txt', encoding='utf-8') as f:
        data_list = f.readlines()
    return data_list


if __name__ == "__main__":
    print('1^ A浙江杭州阿里^AB-prov E-prov B-city E-city B-poi E-poi')
    sentence_list = main()
    print(len(sentence_list))

    final_test = []
    with open('dataset/final_test.txt', encoding='utf-8') as f:
        final_test = f.readlines()
    test_data = []
    print(f'{len(final_test)}\t\t{len(sentence_list)}')
    for i in range(len(final_test)):
        # test_data.append(final_test[i].strip('\n') + '\001' + sentence_list[i] + '\n')
        test_data.append(final_test[i].strip('\n').strip(' ') + '\001' + sentence_list[i].strip(' '))
    with open('predict.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    print(50 * '*')
    print('write result ok!')
    print(50 * '*')

```

    1^ A浙江杭州阿里^AB-prov E-prov B-city E-city B-poi E-poi
    50000
    50000		50000
    **************************************************
    write result ok!
    **************************************************



```python
!head predict.txt
```

    1朝阳区小关北里000-0号B-district I-district E-district B-road I-road I-road E-road B-roadno I-roadno I-roadno I-roadno I-roadno E-roadno
    2朝阳区惠新东街00号B-district I-district E-district B-road I-road I-road E-road B-roadno I-roadno E-roadno
    3朝阳区南磨房路与西大望路交口东南角B-district I-district E-district B-road I-road I-road E-road O B-road I-road I-road E-road B-intersection E-intersection B-assist I-assist E-assist
    4朝阳区潘家园南里00号B-district I-district E-district B-poi I-poi E-poi B-road E-road B-houseno I-houseno E-houseno
    5朝阳区向军南里二巷0号附近B-district I-district E-district B-road I-road I-road E-road B-road E-road B-roadno E-roadno B-assist E-assist
    6朝阳区多处营业网点B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi
    7朝阳区多处营业网点B-district I-district E-district B-poi I-poi B-poi I-poi I-poi E-poi
    8朝阳区多处营业网点B-district I-district E-district B-poi I-poi I-poi I-poi I-poi E-poi
    9朝阳区北三环中路00号商房大厦0楼B-district I-district E-district B-road I-road I-road I-road E-road B-roadno I-roadno E-roadno B-poi I-poi I-poi E-poi B-floorno E-floorno
    10朝阳区孙河乡康营家园00区北侧底商B-district I-district E-district B-town I-town E-town B-poi I-poi I-poi E-poi B-subpoi I-subpoi E-subpoi B-assist E-assist O E-subpoi


## 5.提交格式检查


```python
import linecache


def check(submit_path, test_path, max_num=50000):
    '''
    :param submit_path: 选手提交的文件名
    :param test_path: 原始测试数据名
    :param max_num: 测试数据大小
    :return:
    '''
    N = 0
    with open(submit_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line == '':
                continue
            N += 1
            parts = line.split('\001')  # id, sent, tags
            if len(parts) != 3:
                raise AssertionError(f"分隔符不正确，写入文件时请用'\\001'来分隔ID，句子和预测标签！Error Line:{line.strip()}")
            elif len(parts[1]) != len(parts[2].split(' ')):
                print(line)
                raise AssertionError(f"请保证句子长度和标签长度一致，且标签之间用空格分隔！ID:{parts[0]} Sent:{parts[1]}")
            elif parts[0] != str(N):
                raise AssertionError(f"请保证测试数据的ID合法！ID:{parts[0]} Sent:{parts[1]}")
            else:
                for tag in parts[2].split(' '):
                    if (tag == 'O' or tag.startswith('S-')
                        or tag.startswith('B-')
                        or tag.startswith('I-')
                        or tag.startswith('E-')) is False:
                        raise AssertionError(f"预测结果存在不合法的标签！ID:{parts[0]} Tag:{parts[2]}")

                test_line = linecache.getline(test_path, int(parts[0]))
                test_sent = test_line.strip().split('\001')[1]
                if test_sent.strip() != parts[1].strip():
                    raise AssertionError(f"请不要改变测试数据原文！ID:{parts[0]} Sent:{parts[1]}")

    if N != max_num:
        raise AssertionError(f"请保证测试数据的完整性(共{max_num}条)，不可丢失或增加数据！")

    print('Well Done ！！')


check('predict.txt', 'dataset/final_test.txt')

```

    Well Done ！！


# 七、终于提交成功了
![](https://ai-studio-static-online.cdn.bcebos.com/e1cb12b56376403bba2bd3aecef6c10258d9a05817c94326aed836e0ce926685)


