import argparse

# 创建ArgumentParser对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--device', default='0,1', type=str, required=False, help='设置使用哪些显卡')
parser.add_argument('--epochs', default=10, type=int, required=False, help='训练循环')
parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
parser.add_argument('--output_dir', default='new_model', type=str, required=False, help='模型输出路径')
parser.add_argument('--pretrained_model', default='./model', type=str, required=False, help='模型训练起点路径')

