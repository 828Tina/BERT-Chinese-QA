import torch
import json
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./data/cmrc2018_train.json", encoding="utf-8") as f:
    train = json.load(f)

with open("./data/cmrc2018_dev.json", encoding="utf-8") as f:
    dev = json.load(f)

# 数据处理：
paragraphs = []  # 存储所有段落的内容
questions = []  # 存储所有问题的内容
start_positions = []  # 存储所有答案的开始位置
end_positions = []  # 存储所有答案的结束位置
for paragraph in train["data"]:  # 遍历训练数据的每个段落
    for qa in paragraph["paragraphs"][0]["qas"]:  # 遍历每个段落中的所有问答
        ### START CODE HERE ###
        # 对于每个问题，将其所在段落、问题内容、答案开始位置和结束位置（计算得出）添加到相应的列表中
        paragraphs.append(paragraph["paragraphs"][0]["context"])
        questions.append(qa["question"])
        start_position = qa["answers"][0]["answer_start"]
        start_positions.append(start_position)
        answer_length = len(qa["answers"][0]["text"])
        end_positions.append(start_position + answer_length)

# 导入分词器
tokenizer = BertTokenizerFast.from_pretrained("./model")

train_encodings = tokenizer(
    paragraphs,
    questions,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
)
# `char_to_token` will convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph
train_encodings["start_positions"] = torch.tensor(
    [
        train_encodings.char_to_token(idx, x)
        if train_encodings.char_to_token(idx, x) != None
        else -1
        for idx, x in enumerate(start_positions)
    ]
)
train_encodings["end_positions"] = torch.tensor(
    [
        train_encodings.char_to_token(idx, x - 1)
        if train_encodings.char_to_token(idx, x - 1) != None
        else -1
        for idx, x in enumerate(end_positions)
    ]
)


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: v[idx].to(device) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
