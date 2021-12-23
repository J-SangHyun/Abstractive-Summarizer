# -*- coding: utf-8 -*-
import sentencepiece as spm
from tokenizer.tokenizer import Tokenizer


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_file='./tokenizer/sentencepiece/sentencepiece.model'):
        super(SentencePieceTokenizer, self).__init__(name='sentencepiece')
        self.model_file = model_file

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_file=model_file)
        self.vocab_size = self.sp.vocab_size()

        self.CLS_token = self.sp['[CLS]']
        self.SEP_token = self.sp['[SEP]']
        self.PAD_token = self.sp['[PAD]']
        self.UNK_token = self.sp['[UNK]']

    def tokenize(self, sentence):
        return self.sp.EncodeAsPieces(sentence)

    def encode(self, sentence, max_length):
        indexes = [self.sp[word] for word in self.tokenize(sentence)]
        indexes = [self.CLS_token] + indexes[:max_length - 2] + [self.SEP_token]
        return indexes + [self.PAD_token] * (max_length - len(indexes))

    def decode(self, indexes):
        truncated_indexes = []
        for index in indexes:
            if index == self.CLS_token:
                continue
            elif index == self.SEP_token:
                break
            else:
                truncated_indexes.append(index)
        return self.sp.DecodeIds(truncated_indexes)
