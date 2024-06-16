from tqdm import tqdm
import swanlab
from data import SquadDataset, train_encodings
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
import swanlab
from settings import parser
import os
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

# 设置参数
args = parser.parse_args()

# 加载模型
os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
model = BertForQuestionAnswering.from_pretrained(args.pretrained_model)
# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 如果有多个 GPU，则使用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# 将模型移动到设备
model.to(device)


# 读取数据
train_dataset = SquadDataset(train_encodings)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# 优化器
optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

# 可视化部署
swanlab.init(
    project="Bert_fine-tuning",
    experiment_name="epoch-5",
    workspace=None,
    description="基于BERT的问答模型",
    config={'epochs': args.epochs, 'learning_rate': args.lr},  # 通过config参数保存输入或超参数
    logdir="./logs",  # 指定日志文件的保存路径
)

### 训练
with open('bert_qa.txt', 'a') as f:
    for epoch in range(swanlab.config.epochs):
        model.train()
        loss_sum = 0.0
        acc_start_sum = 0.0
        acc_end_sum = 0.0

        # 只加载一个 batch 的数据
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{swanlab.config.epochs}")):
            # 清除梯度
            optim.zero_grad()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            """
            outputs
            loss:若在单个 GPU 上训练，则它是一个标量；若在多个 GPU 上训练，则它是一个向量，我们需要将它求平均，得到一个标量
            start_logits:这个位置最有可能是 answer 的起始位置,维度为 [batch_size, sequence_length]
            end_logits:这个位置最有可能是 answer 的结束位置,维度为 [batch_size, sequence_length]
            """

            loss = outputs.loss.mean()

            # loss反向传播，更新模型参数
            loss.backward()
            optim.step()

            # loss累加
            loss_sum += loss.item()

            # 计算准确度
            start_pred = torch.argmax(outputs.start_logits, dim=1)
            end_pred = torch.argmax(outputs.end_logits, dim=1)
            acc_start = (start_pred == start_positions).float().mean()
            acc_end = (end_pred == end_positions).float().mean()
            acc_start_sum += acc_start
            acc_end_sum += acc_end

            if batch_idx % 20 == 0 or batch_idx == 1:
                # 将当前批次的统计信息写入文件
                f.write(
                    f"Loss:{loss_sum / (batch_idx + 1):.4f}\t\tAccuracy_start:{acc_start_sum / (batch_idx + 1):.4f}\t\tAccuracy_end:{acc_end_sum / (batch_idx + 1):.4f}")
                swanlab.log({"loss": loss_sum / (batch_idx + 1), "accuracy_start": acc_start_sum / (batch_idx + 1),
                             "accuracy_end": acc_end_sum / (batch_idx + 1)})
        # 每个 epoch 结束后保存模型
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 如果使用 DataParallel, 需要访问 `model.module`
        # 对于单 GPU 训练的模型，直接用 .save_pretrained()
        # model.save_pretrained("./fengchao-bert-qa", from_pt=True)
        # 对于多 GPU 训练得到的模型，要加上 .module
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir, from_pt=True)


