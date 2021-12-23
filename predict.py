# -*- coding: utf-8 -*-
from argparse import ArgumentParser

# Import tokenizer
from tokenizer.okt.okt import OktTokenizer
from tokenizer.sentencepiece.sentencepiece import SentencePieceTokenizer

# Import seq2seq module
from seq2seq.rnn.module import RNNModule
from seq2seq.transformer.module import TransformerModule

# Import global variables
from utils.global_variables import input_max_length, target_max_length


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--tokenizer', type=str, default='sentencepiece', help='choose tokenizer')
    parser.add_argument('-s', '--seq2seq', type=str, default='transformer', help='choose seq2seq model')
    args = parser.parse_args()

    # Set tokenizer
    if args.tokenizer == 'okt':
        tokenizer = OktTokenizer()
    elif args.tokenizer == 'sentencepiece':
        tokenizer = SentencePieceTokenizer()
    else:
        raise NameError(f'tokenizer name {args.tokenizer} is not defined.')
    print(f'Tokenizer Model: {args.tokenizer}')

    # Set seq2seq module
    if args.seq2seq == 'rnn':
        module = RNNModule(tokenizer)
    elif args.seq2seq == 'transformer':
        module = TransformerModule(tokenizer)
    else:
        raise NameError(f'seq2seq model name {args.seq2seq} is not defined.')
    print(f'Seq2seq Model: {args.seq2seq}')

    while True:
        sentence = input('Input: ')
        indexes = tokenizer.encode(sentence, max_length=input_max_length)
        predicted_indexes = module.predict(indexes, max_length=target_max_length)
        print('Output:', tokenizer.decode(predicted_indexes))
        print()
