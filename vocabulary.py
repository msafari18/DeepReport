from collections import Counter
import pickle
import os
import nltk
import time

class Vocabulary(object):

    def __init__(self,
                 vocab_threshold=5,
                 vocab_file='DeepReport_model/vocab.pkl',
                 start_word="<start>",
                 end_word="<end>",
                 pad_word="<pad>",
                 unk_word="<unk>",
                 ign_word="___",
                 report_folder="",
                 train_ids="",
                 test_ids="",
                 vocab_from_file=True):
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.pad_word = pad_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.ign_word = ign_word
        self.vocab_from_file = vocab_from_file
        if not vocab_from_file:
            self.report_folder = report_folder
            with open(train_ids) as file:
                train_ids = [line.strip() for line in file]
            with open(test_ids) as file:
                test_ids = [line.strip() for line in file]
            self.ids = train_ids + test_ids
        self.get_vocab()

    def get_vocab(self):

        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                time.sleep(2)
                vocab = pickle.load(f)
                time.sleep(10)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.pad_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    @staticmethod
    def tokenize(report):
        report = report.lower()
        findings = report.split('findings:')[1].split('impression:')[0]
        tokens = nltk.tokenize.word_tokenize(findings)
        if len(tokens) > 0:
            return tokens
        impression = report.split('impression:')[-1]
        tokens = nltk.tokenize.word_tokenize(impression)
        return tokens

    def add_captions(self):
        counter = Counter()
        for i, id in enumerate(self.ids):
            path = os.path.join(self.report_folder, id) + '.txt'
            with open(path, 'r') as f:
                content = f.read().lower()
            tokens = self.tokenize(content)
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(self.ids)))

        words = [word for word, cnt in counter.items(
        ) if cnt >= self.vocab_threshold and self.ign_word not in word]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
