# -*- coding: utf-8 -*-
import time
import pandas as pd
from argparse import ArgumentParser

# Import tokenizer
from tokenizer.okt.okt import OktTokenizer
from tokenizer.sentencepiece.sentencepiece import SentencePieceTokenizer

# Import seq2seq module
from seq2seq.rnn.module import RNNModule
from seq2seq.transformer.module import TransformerModule

# Import global variables
from utils.global_variables import input_max_length, target_max_length

# Import BLEU score script
from utils.bleu import bleu_wrapper


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--tokenizer', type=str, default='sentencepiece', help='choose tokenizer')
    parser.add_argument('-s', '--seq2seq', type=str, default='transformer', help='choose seq2seq model')
    parser.add_argument('-e', '--eval', type=str,
                        default='./dataset/ncsoft/2020.09.18/eval.csv', help='choose eval csv file')
    parser.add_argument('-a', '--answer', type=str,
                        default='./dataset/ncsoft/2020.09.18/answer.txt', help='choose answer file')
    parser.add_argument('-b', '--batch', type=int, default=256, help='batch size')
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

    predicted_file = './predict.txt'
    eval_corpus = pd.read_csv(args.eval, encoding='utf-8')
    input_sentences = list(eval_corpus['inputs'])
    input_indexes_list = [tokenizer.encode(s, max_length=input_max_length) for s in input_sentences]

    start = time.time()
    predicted_indexes_list = module.predict_batch(indexes_list=input_indexes_list,
                                                  max_length=target_max_length,
                                                  batch=args.batch)

    predicted_file_object = open(predicted_file, 'w', encoding='utf-8')
    for predicted_indexes in predicted_indexes_list:
        predicted_sentence = tokenizer.decode(predicted_indexes)
        predicted_file_object.write(predicted_sentence + '\n')
    predicted_file_object.close()
    time_elapsed = time.time() - start
    print(f'Total Elapsed Time: {time_elapsed}')
    print(f'Elapsed Time per Sentence: {time_elapsed / len(input_sentences)}')

    bleu = bleu_wrapper(ref_filename=args.answer, hyp_filename='./predict.txt')
    print(f'BLEU score: {bleu}')
