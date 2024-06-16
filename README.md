# BERT-Chinese-QA
> 基于BERT的中文问答微调训练

[![swanlab](https://img.shields.io/badge/BERT-SwanLab-438440)](https://swanlab.cn/@LiXinYu/Bert_fine-tuning/runs/ad9o8bbwrmy55lcg3f2y9/chart)

## 环境安装

需要安装以下内容：

```
torch
transformers
swanlab
```

> 本文的代码测试于torch==2.2.2、transformers==4.40.0、swanlab==0.3.0，更多库版本可查看[SwanLab记录的Python环境](https://swanlab.cn/@LiXinYu/Bert_fine-tuning/runs/fys4kohptcdt3odf9l3jt/environment/requirements)


## 加载模型
BERT模型我们直接下载来自HuggingFace上由Google发布的[bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)预训练模型。


当然也可以直接执行下面的代码，会自动下载模型权重并加载模型：

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese')
```

## 下载数据集

本次训练使用的是CMRC2018数据集，[CMRC2018](https://hfl-rc.github.io/cmrc2018/)是第二届[讯飞杯]中文机器阅读理解颁奖研讨会中相关赛事所使用的资料集，它主要用于中文机器阅读理解的跨度提取资料集，以增加该领域的语言多样性.

该资料集有人类专家在维基百科段落上注释的近20000个真实问题组成。

同时也注释了一个挑战集，其中包含需要在整个上下文中进行全面理解和多句推理的问题。

代码链接：[ymcui/cmrc2018](https://github.com/ymcui/cmrc2018)

```python

# 加载数据集
with open("./data/cmrc2018_train.json", encoding="utf-8") as f:
    train = json.load(f)

with open("./data/cmrc2018_dev.json", encoding="utf-8") as f:
    dev = json.load(f)

```

## 使用swanlab可视化结果

```python
# 可视化部署
swanlab.init(
    project="Bert_fine-tuning",
    experiment_name="epoch-5",
    workspace=None,
    description="基于BERT的问答模型",
    config={'epochs': args.epochs, 'learning_rate': args.lr},  # 通过config参数保存输入或超参数
    logdir="./logs",  # 指定日志文件的保存路径
)
```

想了解更多关于SwanLab的知识，请看[SwanLab官方文档](https://docs.swanlab.cn/zh/guide_cloud/general/what-is-swanlab.html)。

## 训练

训练过程可视化：[BERT-QA-Swanlab](https://swanlab.cn/@LiXinYu/Bert_fine-tuning/runs/ad9o8bbwrmy55lcg3f2y9/chart)

在首次使用SwanLab时，需要去[官网](https://swanlab.cn)注册一下账号，然后在[用户设置](https://swanlab.cn/settings)复制一下你的API Key。

然后在终端输入`swanlab login`:

```bash
swanlab login
```

把API Key粘贴进去即可完成登录，之后就不需要再次登录了。






