import os
import h5py
import torch
import random
from torch.utils.data import Dataset
from DeepReport_model.vocabulary import Vocabulary
from string import punctuation


def join_words(b):
    punc = set(punctuation)
    return ''.join(w if set(w) <= punc else ' ' + w for w in b).lstrip()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f'avg:{self.avg:.3f}'


class FeatReportBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.feats = torch.stack(transposed_data[0], 0)
        self.reports = transposed_data[1]
        self.reports_len = [len(r) for r in self.reports]


class ReportGenerationDataset(Dataset):
    def __init__(self, features_h5_path, report_dir, id_path='./data/test.txt',  vocab_file='./data/vocab.pkl', max_len=400, pad=False):
        self.features_h5_path = features_h5_path
        self.report_dir = report_dir
        self.max_len = max_len
        self.vocab = Vocabulary(vocab_file=vocab_file)
        self.pad = pad
        with open(id_path, 'r') as f:
            self.ids = [id.strip() for id in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        with h5py.File(self.features_h5_path, "r") as feature_h5:
            patient_id = self.ids[idx]
            dicoms = list(feature_h5[patient_id])
            dicom = random.choice(dicoms)
            feature = torch.Tensor(feature_h5[patient_id + '/' + dicom][:])

        with open(os.path.join(self.report_dir, patient_id + '.txt'), 'r') as f:
            content = f.read().lower()

        tokens = []
        tokenized = self.vocab.tokenize(content)[:self.max_len - 2]
        tokens.append(self.vocab(self.vocab.start_word))
        tokens.extend([self.vocab(t) for t in tokenized])
        tokens.append(self.vocab(self.vocab.end_word))
        if self.pad:
            tokens = tokens + \
                [self.vocab(self.vocab.pad_word)] * \
                (self.max_len - len(tokens))

        tokens = torch.Tensor(tokens).long()
        return feature, tokens
