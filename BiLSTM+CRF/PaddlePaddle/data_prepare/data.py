import paddle
import paddle.nn as nn

import paddlenlp
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator


class Data:
    def __init__(self, word_dic, label_dic, train_path, dev_path, test_path):
        self.word_vocab = None
        self.label_vocab = None
        self.word_dic = word_dic
        self.label_dic = label_dic
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

    # 将每一个字对应一个数字
    def load_dict(self, dict_path):
        vocab = {}
        i = 0
        for line in open(dict_path, "r", encoding="utf-8"):
            key = line.strip("\n")
            vocab[key] = i
            i += 1
        return vocab

    def set_word_vocab(self):
        self.word_vocab = self.load_dict(self.word_dic)

    def set_label_vocab(self):
        self.label_vocab = self.load_dict(self.label_dic)

    def get_word_vocab(self):
        return self.word_vocab

    def get_label_vocab(self):
        return self.label_vocab

    def convert_tokens_to_ids(self, tokens, vocab, oov_token=None):
        token_ids = []
        oov_id = vocab.get(oov_token) if oov_token else None
        for token in tokens:
            token_id = vocab.get(token, oov_id)
            token_ids.append(token_id)
        return token_ids

    def load_dataset(self, datafiles):
        def read(data_path):
            with open(data_path, 'r', encoding="utf-8") as fp:
                next(fp)  # 跳过第一行
                for line in fp.readlines():
                    words, labels = line.strip('\n').split("\t")
                    words = words.split("\002")
                    labels = labels.split("\002")
                    yield words, labels

        if isinstance(datafiles, str):
            return MapDataset(list(read(datafiles)))
        elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
            return [MapDataset(list(read(datafile))) for datafile in datafiles]

    def convert_example(self, example):
        tokens, labels = example
        token_ids = self.convert_tokens_to_ids(tokens, self.word_vocab, 'OOV')
        label_ids = self.convert_tokens_to_ids(labels, self.label_vocab, 'O')
        return token_ids, len(token_ids), label_ids

    def get_loaders(self):
        train_ds, dev_ds, test_ds = self.load_dataset(
            datafiles=(self.train_path, self.dev_path, self.test_path))

        train_ds.map(self.convert_example)
        dev_ds.map(self.convert_example)
        test_ds.map(self.convert_example)


        batchify_fn = lambda samples, fn=Tuple(Pad(axis=0, pad_val=self.word_vocab.get('OOV')),
                                               Stack(dtype='int32'),
                                               Pad(axis=0, pad_val=self.label_vocab.get('O'))): fn(samples)

        train_loader = paddle.io.DataLoader(dataset=train_ds, batch_size=32, shuffle=True, drop_last=True, return_list=True,
                                            collate_fn=batchify_fn)
        dev_loader = paddle.io.DataLoader(dataset=dev_ds, batch_size=32, drop_last=True, return_list=True,
                                          collate_fn=batchify_fn)
        test_loader = paddle.io.DataLoader(dataset=test_ds, batch_size=32, drop_last=True, return_list=True,
                                           collate_fn=batchify_fn)
        return train_loader, dev_loader, test_loader


















