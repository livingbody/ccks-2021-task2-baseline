# 一、“英特尔创新大师杯”深度学习挑战赛 赛道3：CCKS2021中文NLP地址相关性任务

赛题地址： [https://tianchi.aliyun.com/competition/entrance/531901/information](https://tianchi.aliyun.com/competition/entrance/531901/information)

aistudio地址: [https://aistudio.baidu.com/aistudio/projectdetail/2158092](https://aistudio.baidu.com/aistudio/projectdetail/2158092)

## 1.赛题背景
地址文本相关性任务在现实世界中存在着广泛的应用场景，如：基于地理信息搜索的地理位置服务、对于突发事件位置信息的快速搜索定位、不同地址信息系统的对齐等等。

日常生活中输入的地址文本可以为以下几种形式：

* 包含四级行政区划及路名路号POI的规范地址文本；
* 地址要素缺省的规范地址文本，例：只有路名+路号、只有POI；
* 非规范的地址文本、口语化的地址信息描述，例：阿里西溪园区东门旁亲橙里；
* 地址文本相关性主要是衡量地址间的相似程度，地址要素解析与地址相关性共同构成了中文地址处理两大核心任务，具有很大的商业价值。目前中文地址领域缺少标准的评测和数据集，这次我们将开放较大规模的标注语料，希望和社区共同推动地址文本处理领域的发展。

## 2.赛题描述
本评测任务为基于地址文本的相关性任务。即对于给定的一个地址query以及若干个候选地址文本，参赛系统需要对query与候选地址文本的匹配程度进行打分。

多样化的地址文本写法对地址文本的相关性任务提出的挑战如下：

* 同一个地址存在多种写法，没有给定的改写词表；
* 地址query一般存在省市区等限制条件，需要结合限制条件分析相关性；
* 不同地市地址规范不一，对模型泛化性提出更高要求；
## 3.数据说明
输入：输入文件包含若干个query-地址文本对

输出：输出文本每一行包括此query-地址文本对的匹配程度，分为完全匹配、部分匹配、不匹配

# 二、数据处理

## 1.paddlenlp更新


```python
!pip install paddlenlp
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: paddlenlp in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (2.0.1)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.2.0)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied: numpy>=1.7 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from h5py->paddlenlp) (1.20.3)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from h5py->paddlenlp) (1.15.0)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.2.3)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.5)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.24.2)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.4.2)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.10.1)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.23)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.6.3)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (2.1.0)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl->paddlenlp) (56.2.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (7.2.0)


## 2.数据解压缩


```python
# !unzip 'data/data95669/“英特尔创新大师杯”深度学习挑战赛 赛道3：CCKS2021中文NLP地址相关性任务.zip'
```


```python
# !mv 'б░╙в╠╪╢√┤┤╨┬┤є╩ж▒нб▒╔ю╢╚╤з╧░╠Ї╒╜╚№ ╚№╡└3г║CCKS2021╓╨╬─NLP╡╪╓╖╧р╣╪╨╘╚╬╬ё' dataset
```


```python
# !mv 'dataset/╕ё╩╜╫╘▓щ╜┼▒╛.py' check.py
```


```python
!head dataset/Xeon3NLP_round1_train_20210524.txt
```

    {"text_id": "e225b9fd36b8914f42c188fc92e8918f", "query": "河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元", "candidate": [{"text": "巩义市桐和街", "label": "不匹配"}, {"text": "桐和街依家小店", "label": "不匹配"}, {"text": "桐和街CHANG六LIULIU", "label": "不匹配"}, {"text": "桐和街佳乐钢琴", "label": "不匹配"}, {"text": "世博领秀城南门桐和街囍饭食堂", "label": "不匹配"}]}
    {"text_id": "b2418ead7b48db4c09caa2934843c1b4", "query": "老垅坡高家组省建五公司", "candidate": [{"text": "高家巷省建五公司岳阳分公司", "label": "完全匹配"}, {"text": "建设北路346号省建5公司", "label": "不匹配"}, {"text": "老垅坡路西100米省建三公司(岳阳分公司)", "label": "不匹配"}, {"text": "寇庄西路101号省建五公司", "label": "不匹配"}, {"text": "卓刀泉南路21号省五建公司省建五公司", "label": "不匹配"}]}
    {"text_id": "5fa94565eb53463fa94ece56e8356fdc", "query": "西关人和路工商银行对过腾信医药一楼眼镜店", "candidate": [{"text": "河门口北街33号中国工商银行(河门口支行)", "label": "不匹配"}, {"text": "河门口北街33号中国工商银行ATM(河门口支行)", "label": "不匹配"}, {"text": "清香坪东街(食为天旁)中国工商银行24小时自助银行", "label": "不匹配"}, {"text": "陶家渡东路209号中国工商银行24小时自助银行(巴关河支行)", "label": "不匹配"}, {"text": "苏铁中路110号中国工商银行24小时自助银行(清香坪支行)", "label": "不匹配"}]}
    {"text_id": "10a2a7c833eea18f479a58f2d15a53b5", "query": "唐海县四农场场部王玉文", "candidate": [{"text": "场前路北50米曹妃甸区第四农场", "label": "部分匹配"}, {"text": "新区曹妃甸湿地曹妃湖东北侧曹妃甸慧钜文化创意产业园", "label": "不匹配"}, {"text": "建设大街255号四季华庭", "label": "不匹配"}, {"text": "曹妃甸区西环路", "label": "不匹配"}, {"text": "华兴路4附近曹妃歌厅(西门)", "label": "不匹配"}]}
    {"text_id": "7b82872ddc84f94733f5dac408d98bce", "query": "真北路818号近铁城市广场北座二楼", "candidate": [{"text": "真北路818号近铁城市广场北座", "label": "部分匹配"}, {"text": "真北路818号近铁城市广场北座(西南2门)", "label": "部分匹配"}, {"text": "真北路818号近铁城市广场北座(西南1门)", "label": "部分匹配"}, {"text": "真北路818号近铁城市广场北座2层捞王火锅", "label": "部分匹配"}, {"text": "金沙江路1685号118广场F1近铁城市广场北座(西北门)", "label": "部分匹配"}]}
    {"text_id": "24dbf73a46d68ef94bd69c8728adeed6", "query": "义亭工业区甘塘西路9号", "candidate": [{"text": "义亭镇甘塘西路9号秀颜化妆用具有限公司", "label": "部分匹配"}, {"text": "义亭镇甘塘西路9-1号义乌市恒凯玩具有限公司", "label": "部分匹配"}, {"text": "义乌市甘塘西路", "label": "不匹配"}, {"text": "黄金塘西路9号丹阳英福康电子科技有限公司", "label": "不匹配"}, {"text": "黄塘西路9号二楼卓悦国际舞蹈学院", "label": "不匹配"}]}
    {"text_id": "a48b64238490d0447bd5b39e527b280f", "query": "溧水县永阳镇东山大队谢岗村31号", "candidate": [{"text": "溧水区谢岗", "label": "不匹配"}, {"text": "溧水区东山", "label": "部分匹配"}, {"text": "永阳镇中山大队", "label": "不匹配"}, {"text": "溧水区东山线", "label": "不匹配"}, {"text": "东山线附近东山林业队", "label": "不匹配"}]}
    {"text_id": "adb0c58f99a60ae136daa5148fa188a7", "query": "纪家庙万兴家居D座二楼", "candidate": [{"text": "南三环西路玉泉营万兴家居D座2层德高防水", "label": "部分匹配"}, {"text": "玉泉营桥西万兴家居D座二层德国都芳漆", "label": "部分匹配"}, {"text": "花乡南三环玉泉营桥万兴家居D管4层羽翔红木", "label": "部分匹配"}, {"text": "花乡玉泉营桥西万兴家居D座3层69-71五洲装饰", "label": "部分匹配"}, {"text": "花乡乡南三环西路78号万兴国际家居广场纪家庙地区万兴家居广场", "label": "部分匹配"}]}
    {"text_id": "78f05d3265d15acde1e98cda41cf4f12", "query": "江苏省南京市江宁区禄口街道欢墩山", "candidate": [{"text": "江宁区欢墩山", "label": "部分匹配"}]}
    {"text_id": "9601ec765ee5a860992ec83b5743fe42", "query": "博美二期大门对面莲花新区7号地", "candidate": [{"text": "围场满族蒙古族自治县七号地", "label": "不匹配"}, {"text": "金梧桐宾馆东侧100米大屯7号地", "label": "不匹配"}, {"text": "中原路商隐路交汇处清华·大溪地七号院", "label": "不匹配"}, {"text": "滨海新区博美园7号楼", "label": "不匹配"}, {"text": "莲池区秀兰城市美地7号楼", "label": "不匹配"}]}


## 3.paddlenlp库引入


```python
# paddlenlp库引入
import paddlenlp as ppnlp
import paddle.nn.functional as F

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.transformers import LinearDecayWithWarmup

import numpy as np
import os
import json
import time
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp

from functools import partial
```

## 4.自定义reader


```python
train_tmp = []
dev_tmp = []
num = 0 
a = []
for line in open('dataset/Xeon3NLP_round1_train_20210524.txt','r'):
    t = json.loads(line)
    for j in t['candidate']:
        l = dict()
        l['query'] = str(t['query'])
        l['title'] = str(j['text'])
        a.append(len(l['title']))
        if j['label'] == '不匹配':
            l['label'] = 0
        elif j['label'] == '完全匹配':
            l['label'] = 2
        else:
            l['label'] = 1
        if num <18000:
            train_tmp.append(l)
        else:
            dev_tmp.append(l)
    num += 1
num
```




    20000




```python
from paddlenlp.datasets import load_dataset
def read(filename):
    for line in filename:
        yield {'query': line['query'], 'title': line['title'], 'label':line['label']}
```


```python
train_ds = load_dataset(read, filename=train_tmp, lazy=False)
dev_ds = load_dataset(read, filename=dev_tmp, lazy=False)
```

## 5.数据预处理


```python
for idx, example in enumerate(train_ds):
    if idx <= 5:
        print(example)
        
def convert_example(example, tokenizer, max_seq_length=80, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
pretrained_model = paddlenlp.transformers.RobertaModel.from_pretrained('roberta-wwm-ext')

tokenizer = paddlenlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext')

```

    {'query': '河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元', 'title': '巩义市桐和街', 'label': 0}
    {'query': '河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元', 'title': '桐和街依家小店', 'label': 0}
    {'query': '河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元', 'title': '桐和街CHANG六LIULIU', 'label': 0}
    {'query': '河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元', 'title': '桐和街佳乐钢琴', 'label': 0}
    {'query': '河南省巩义市新华路街道办事处桐和街6号钢苑新区3号楼一单元', 'title': '世博领秀城南门桐和街囍饭食堂', 'label': 0}
    {'query': '老垅坡高家组省建五公司', 'title': '高家巷省建五公司岳阳分公司', 'label': 2}


    [2021-07-07 01:20:02,439] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/roberta-wwm-ext/roberta_chn_base.pdparams
    [2021-07-07 01:20:07,198] [    INFO] - Found /home/aistudio/.paddlenlp/models/roberta-wwm-ext/vocab.txt


# 三、模型训练

## 1.构建模型


```python
import paddle.nn as nn

class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE1.0 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 3)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        return logits
```


```python
model = PointwiseMatching(pretrained_model)
```

## 2.定义样本转换函数


```python
# 将 1 条明文数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据
# 返回 input_ids 和 token_type_ids

def convert_example(example, tokenizer, max_seq_length=80, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
```


```python
# 为了后续方便使用，我们给 convert_example 赋予一些默认参数
from functools import partial

# 训练集和验证集的样本转换函数
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=80)
    # 512
```

## 3. 组装 Batch 数据 & Padding


```python

from paddlenlp.data import Stack, Pad, Tuple
# 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
# 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
```

## 4.定义 Dataloader


```python
# 基于 train_ds 定义 train_data_loader
# 因为我们使用了分布式的 DistributedBatchSampler, train_data_loader 会自动对训练数据进行切分
# 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=400, shuffle=True)
train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
# 定义 dev_data_loader
batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=400, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```


```python
batch_size = 16
# 训练过程中的最大学习率
learning_rate = 2e-5 
# 训练轮次
epochs = 100
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 1e-4
```


```python
num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
```

## 5.模型训练


```python
from paddlenlp.transformers import LinearDecayWithWarmup

num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()
```


```python
# 加入日志显示
from visualdl import LogWriter

writer = LogWriter("./log")
```


```python
# 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    
    # 加入eval日志显示
    writer.add_scalar(tag="eval/loss", step=global_step, value=np.mean(losses))
    writer.add_scalar(tag="eval/acc", step=global_step, value=accu)                                                  
    model.train()
    metric.reset()
    return accu
```


```python
# 接下来，开始正式训练模型
save_dir = "checkpoint"
os.makedirs(save_dir)
global_step = 0
tic_train = time.time()
pre_accu=0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):

        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        
        # 每间隔 10 step 输出训练指标
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()


        # 每间隔 100 step 在验证集和测试集上进行评估
        if global_step % 100 == 0:
            accu=evaluate(model, criterion, metric, dev_data_loader, "dev")
        
        # 加入train日志显示
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)
            writer.add_scalar(tag="train/acc", step=global_step, value=acc)
            
            if accu>pre_accu:
                # 加入保存
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                # tokenizer.save_pretrained(save_dir)
                pre_accu=accu
            

```

    global step 15580, epoch: 72, batch: 173, loss: 0.00863, accu: 0.99809, speed: 0.65 step/s


    ---------------------------------------------------------------------------
    
    KeyboardInterrupt                         Traceback (most recent call last)
    
    <ipython-input-22-1b431880b46f> in <module>
          9 
         10         input_ids, token_type_ids, labels = batch
    ---> 11         probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
         12         loss = criterion(probs, labels)
         13         correct = metric.compute(probs, labels)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    <ipython-input-11-ec17a4daab3b> in forward(self, input_ids, token_type_ids, position_ids, attention_mask)
         17 
         18         _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
    ---> 19                                     attention_mask)
         20         cls_embedding = self.dropout(cls_embedding)
         21         logits = self.classifier(cls_embedding)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlenlp/transformers/roberta/modeling.py in forward(self, input_ids, token_type_ids, position_ids, attention_mask)
        332         if attention_mask is None:
        333             attention_mask = paddle.unsqueeze(
    --> 334                 (input_ids == self.pad_token_id
        335                  ).astype(self.pooler.dense.weight.dtype) * -1e9,
        336                 axis=[1, 2])


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py in __impl__(self, other_var)
        248             axis = -1
        249             math_op = getattr(core.ops, op_type)
    --> 250             return math_op(self, other_var, 'axis', axis)
        251 
        252         comment = OpProtoHolder.instance().get_op_proto(op_type).comment


    KeyboardInterrupt: 


### 训练日志截取
```
global step 510, epoch: 3, batch: 76, loss: 0.31065, accu: 0.87175, speed: 0.34 step/s
global step 520, epoch: 3, batch: 86, loss: 0.36163, accu: 0.87113, speed: 0.64 step/s
global step 530, epoch: 3, batch: 96, loss: 0.27322, accu: 0.87517, speed: 0.64 step/s
global step 540, epoch: 3, batch: 106, loss: 0.38036, accu: 0.87538, speed: 0.65 step/s
global step 550, epoch: 3, batch: 116, loss: 0.30168, accu: 0.87545, speed: 0.66 step/s
global step 560, epoch: 3, batch: 126, loss: 0.34728, accu: 0.87667, speed: 0.66 step/s
global step 570, epoch: 3, batch: 136, loss: 0.28104, accu: 0.87754, speed: 0.65 step/s
global step 580, epoch: 3, batch: 146, loss: 0.30536, accu: 0.87803, speed: 0.68 step/s
global step 590, epoch: 3, batch: 156, loss: 0.35877, accu: 0.87739, speed: 0.65 step/s
global step 600, epoch: 3, batch: 166, loss: 0.39249, accu: 0.87828, speed: 0.65 step/s
eval dev loss: 0.47208, accu: 0.83651
```

## 6.模型保存


```python
# 训练结束后，存储模型参数
save_dir = os.path.join("checkpoint2", "model_%d" % global_step)
os.makedirs(save_dir)

save_param_path = os.path.join(save_dir, 'model_state111.pdparams')
paddle.save(model.state_dict(), save_param_path)
tokenizer.save_pretrained(save_dir)
```

# 四、预测

## 1.载入模型
* 训练完毕后先重启释放缓存
* 再运行上面训练前的代码


```python
model_dict=paddle.load('checkpoint/model_state.pdparams')
model.set_dict(model_dict)
```


```python
batchify_test_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
): [data for data in fn(samples)]
```

##  2.开始训练


```python
num = 0
fo = open("submit_addr_match_runid4.txt", "w")
for line in open('dataset/Xeon3NLP_round1_test_20210524.txt','r', encoding='utf8'):
    t = json.loads(line)
    for j in range(len(t['candidate'])):
        tmp = []
        l = dict()
        l['query'] = t['query']
        l['title'] = t['candidate'][j]['text']
        l['label'] = 0
        input_ids, token_type_ids,label = convert_example(l, tokenizer, max_seq_length=80)
        tmp.append((input_ids, token_type_ids))
        input_ids, token_type_ids = batchify_test_fn(tmp)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        idx = paddle.argmax(logits, axis=1).numpy()
        if idx[0] == 0:
            t['candidate'][j]['label'] = '不匹配'
        elif idx[0] == 1:
            t['candidate'][j]['label'] = '部分匹配'
        else:
            t['candidate'][j]['label'] = '完全匹配'
    # if num < 10:
    #     print(t)
    # print(t['candidate'][j]['label'] )
    if num % 500 == 0:
        print(num)
    fo.write(json.dumps(t,ensure_ascii=False))
    fo.write('\n')
    num +=1
print(40*'*')
print(f"num: f{num}")
fo.close()
```

    0
    500
    1000
    1500
    2000
    2500
    3000
    3500
    4000
    4500
    5000
    5500
    6000
    6500
    7000
    7500
    8000
    8500
    9000
    9500
    10000
    10500
    11000
    11500
    12000
    12500
    13000
    13500
    14000
    14500
    ****************************************
    num: f15000


## 3.格式检查并提交


```python
import json
import linecache


def check(submit_path, test_path, max_num=15000):
    '''
    :param submit_path: 选手提交的文件名
    :param test_path: 原始测试数据名
    :param max_num: 测试数据大小
    :return:
    '''
    N = 0
    with open(submit_path, 'r', encoding='utf-8') as fs:
        for line in fs:
            try:
                line = line.strip()
                if line == '':
                    continue
                N += 1
                smt_jsl = json.loads(line)
                test_jsl = json.loads(linecache.getline(test_path, N))
                if set(smt_jsl.keys()) != {'text_id', 'query', 'candidate'}:
                    raise AssertionError(f'请保证提交的JSON数据的所有key与测评数据一致！ {line}')
                elif smt_jsl['text_id'] != test_jsl['text_id']:
                    raise AssertionError(f'请保证text_id和测评数据一致，并且不要改变数据顺序！text_id: {smt_jsl["text_id"]}')
                elif smt_jsl['query'] != test_jsl['query']:
                    raise AssertionError(f'请保证query内容和测评数据一致！text_id: {smt_jsl["text_id"]}, query: {smt_jsl["query"]}')
                elif len(smt_jsl['candidate']) != len(test_jsl['candidate']):
                    raise AssertionError(f'请保证candidate的数量和测评数据一致！text_id: {smt_jsl["text_id"]}')
                else:
                    for smt_cand, test_cand in zip(smt_jsl['candidate'], test_jsl['candidate']):
                        if smt_cand['text'] != test_cand['text']:
                            raise AssertionError(f'请保证candidate的内容和顺序与测评数据一致！text_id: {smt_jsl["text_id"]}, candidate_item: {smt_cand["text"]}')

            except json.decoder.JSONDecodeError:
                raise AssertionError(f'请检查提交数据的JSON格式是否有误！如：用键-值用双引号 {line}')
            except KeyError:
                raise AssertionError(f'请保证提交的JSON数据的所有key与测评数据一致！{line}')

    if N != max_num:
        raise AssertionError(f"请保证测试数据的完整性(共{max_num}条)，不可丢失或增加数据！")

    print('Well Done ！！')

check('submit_addr_match_runid4.txt', 'dataset/Xeon3NLP_round1_test_20210524.txt')
```

# 五、成绩
大概2个epoch，成绩约为0.76，希望大家可以取得更好的成绩。

![](https://ai-studio-static-online.cdn.bcebos.com/5b990780570e46bd8b142bd28c97ad76c610433ce23749c497f76ec2428d5220)

