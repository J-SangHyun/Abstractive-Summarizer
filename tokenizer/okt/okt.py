# -*- coding: utf-8 -*-
import pickle
import pandas as pd
from glob import glob
from konlpy.tag import Okt
from tokenizer.tokenizer import Tokenizer


class OktTokenizer(Tokenizer):
    def __init__(self, model_file='./tokenizer/okt/okt.model', train_mode=False):
        super(OktTokenizer, self).__init__(name='okt')
        self.tokenizer = Okt()

        self.CLS_token = 0
        self.SEP_token = 1
        self.PAD_token = 2
        self.UNK_token = 3

        self.word2index = {
            '[CLS]': self.CLS_token,
            '[SEP]': self.SEP_token,
            '[PAD]': self.PAD_token,
            '[UNK]': self.UNK_token
        }
        self.index2word = {
            self.CLS_token: '[CLS]',
            self.SEP_token: '[SEP]',
            self.PAD_token: '[PAD]',
            self.UNK_token: ' ⁇ '
        }
        self.vocab_size = 4

        if not train_mode:
            self.load_word_dict(model_file=model_file)

    def tokenize(self, sentence):
        return self.tokenizer.morphs(sentence)

    def build_word_dict(self, sentences):
        for sentence in sentences:
            tokenized_sentence = self.tokenize(sentence)
            for word in tokenized_sentence:
                if word not in self.word2index:
                    self.word2index[word] = self.vocab_size
                    self.index2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, sentence, max_length):
        indexes = []
        for word in self.tokenize(sentence):
            if word not in self.word2index:
                indexes.append(self.UNK_token)
            else:
                indexes.append(self.word2index[word])

        indexes = [self.CLS_token] + indexes[:max_length - 2] + [self.SEP_token]
        return indexes + [self.PAD_token] * (max_length - len(indexes))

    def decode(self, indexes):
        sentence = ''
        for index in indexes:
            if index == self.CLS_token:
                continue
            elif index == self.SEP_token:
                break
            else:
                sentence += self.index2word[index]
        return sentence

    def save_word_dict(self, model_file):
        with open(model_file, 'wb') as f:
            pickle.dump((self.word2index, self.index2word), f)

    def load_word_dict(self, model_file):
        with open(model_file, 'rb') as f:
            self.word2index, self.index2word = pickle.load(f)


# Build new Okt model using train dataset
if __name__ == '__main__':
    tokenizer = OktTokenizer(train_mode=True)

    input_sentences, target_sentences = [], []
    for train_file in glob('../../dataset/*/*/train.csv'):
        corpus = pd.read_csv(train_file, encoding='utf-8')
        corpus.dropna(axis=0, inplace=True)
        input_sentences += list(corpus['inputs'])
        target_sentences += list(corpus['targets'])

    tokenizer.build_word_dict(input_sentences + target_sentences)
    tokenizer.save_word_dict(model_file='./okt.model')
    print(f'Okt model saved. ({tokenizer.vocab_size} Words)')
