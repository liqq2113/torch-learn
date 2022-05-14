import torch

from data_loader import Data
import argparse


# 参数传输
parser = argparse.ArgumentParser(description="BiLSTM-CRF")
# 运行参数
parser.add_argument('--cuda-able', action='store_true', help='enable cuda')

# 数据参数
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
# 模型参数
parser.add_argument('--embed-dim', type=int, default=256, help='number of embedding dimension [default: 256]')
parser.add_argument('--hidden-dim', type=int, default=256, help='BiLSTM hidden size [default: 256]')
parser.add_argument('--lstm-layers', type=int, default=3, help='BiLSTM layer numbers [default: 3]')
parser.add_argument('--scope', type=float, default=0.25, help='weight init scope')
# 训练参数
parser.add_argument('--dropout', type=float, default=0.3, help='the probability for dropout (0 = no dropout) [default: 0.3]')
args = parser.parse_args()
torch.manual_seed(args.seed)
# cuda 使用

use_cuda = torch.cuda.is_available() and args.cuda_able
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# 第一步：数据处理
data_class = Data("./data/word.dic", "./data/tag.dic", './data/train.txt', './data/dev.txt', './data/test.txt')
data_class.set_word_vocab()
data_class.set_label_vocab()
word_vocab = data_class.get_word_vocab()
label_vocab = data_class.get_label_vocab()
train_loader, dev_loader, test_loader = data_class.get_loaders(args.batch_size)

#第二步：搭建模型
from model import BiLSTM_CRF

model = BiLSTM_CRF(args)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss





