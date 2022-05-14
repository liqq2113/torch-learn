import paddle
from data_prepare.data import Data
from model import BiGRUWithCRF
from paddlenlp.layers import LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator


# 第一步：数据处理
data_class = Data("../data/word.dic", "../data/tag.dic", '../data/train.txt', '../data/dev.txt', '../data/test.txt')
data_class.set_word_vocab()
data_class.set_label_vocab()
word_vocab = data_class.get_word_vocab()
label_vocab = data_class.get_label_vocab()
train_loader, dev_loader, test_loader = data_class.get_loaders()

# 第二步：加载模型
# Define the model netword and its loss
network = BiGRUWithCRF(300, 300, len(word_vocab), len(label_vocab))
model = paddle.Model(network)

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
crf_loss = LinearChainCrfLoss(network.crf)
chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)
model.prepare(optimizer, crf_loss, chunk_evaluator)

# 第三步：模型训练
model.fit(train_data=train_loader, eval_data=dev_loader, epochs=1, save_dir='./results', log_freq=1)

# 网络调试参数
# 数据相关参数