import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class MyFDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


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
            dataset = list(read(datafiles))
            sentences = []
            labels = []
            for data in dataset:
                sentence, label_list = data
                sentences.append(sentence)
                labels.append(label_list)
            return MyDataset(sentences, labels)
        elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
            datasets = []
            for datafile in datafiles:
                dataset = list(read(datafile))
                sentences = []
                labels = []
                for data in dataset:
                    sentence, label_list = data
                    sentences.append(sentence)
                    labels.append(label_list)
                datasets.append(MyDataset(sentences, labels))
            return datasets

    def convert_example(self, example):
        tokens, labels = example
        token_ids = self.convert_tokens_to_ids(tokens, self.word_vocab, 'OOV')
        label_ids = self.convert_tokens_to_ids(labels, self.label_vocab, 'O')
        return token_ids, len(token_ids), label_ids

    def convert_to_idx(self, dataset):
        data, labels = dataset.data, dataset.labels
        data_idx, lengths, label_idx = [], [], []
        for index, idata in enumerate(data):
            data_ids, count, label_ids = self.convert_example((idata, labels[index]))
            data_idx.append(data_ids)
            lengths.append(count)
            label_idx.append(label_ids)
        # 得到句子的最大长度
        max_len = max(lengths)
        # 对data_idx, label_idx 进行补全
        data_idx = self.padding(data_idx, max_len)
        label_idx = self.padding(label_idx, max_len)
        lengths = [max_len for _ in label_idx]
        return data_idx, lengths, label_idx

    def padding(self, seq, max_len):
        pad_seq = np.array([inst + [0]*(max_len-len(inst)) for inst in seq])
        pad_seq = torch.LongTensor(pad_seq)
        return pad_seq

    def get_loaders(self, batch_size):
        train_ds, dev_ds, test_ds = self.load_dataset(
            datafiles=(self.train_path, self.dev_path, self.test_path))
        # 将数据全部转换为数字形式 + 句子补全
        train_ds = self.convert_to_idx(train_ds)
        dev_ds = self.convert_to_idx(dev_ds)
        test_ds = self.convert_to_idx(test_ds)
        # 创建dataset
        train_set = MyFDataset(train_ds[0], train_ds[2])
        dev_set = MyFDataset(dev_ds[0], dev_ds[2])
        test_set = MyFDataset(test_ds[0], test_ds[2])

        train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, drop_last=False)
        dev_loader = DataLoader(dataset=dev_set, batch_size=4, shuffle=True, drop_last=False)
        test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, drop_last=False)







