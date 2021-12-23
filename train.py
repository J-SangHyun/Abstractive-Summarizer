# -*- coding: utf-8 -*-
import time
import datetime
import pandas as pd
from glob import glob
from argparse import ArgumentParser

# Import tokenizer
from tokenizer.okt.okt import OktTokenizer
from tokenizer.sentencepiece.sentencepiece import SentencePieceTokenizer

# Import seq2seq trainer
from seq2seq.rnn.trainer import RNNTrainer
from seq2seq.transformer.trainer import TransformerTrainer

# Import global variables
from utils.global_variables import input_max_length, target_max_length


if __name__ == '__main__':
    # Argument handling
    parser = ArgumentParser()
    parser.add_argument('-t', '--tokenizer', type=str, default='sentencepiece', help='choose tokenizer')
    parser.add_argument('-s', '--seq2seq', type=str, default='transformer', help='choose seq2seq model')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='train epochs')
    parser.add_argument('-b', '--batch', type=int, default=128, help='batch size')
    args = parser.parse_args()

    # Load train data
    input_sentences, target_sentences = [], []
    for train_file in glob('./dataset/*/*/train.csv'):
        corpus = pd.read_csv(train_file, encoding='utf-8')
        corpus.dropna(axis=0, inplace=True)
        input_sentences += list(corpus['inputs'])
        target_sentences += list(corpus['targets'])
    print(f'Valid Data Number: {len(input_sentences)}')

    # Set tokenizer
    if args.tokenizer == 'okt':
        tokenizer = OktTokenizer()
    elif args.tokenizer == 'sentencepiece':
        tokenizer = SentencePieceTokenizer()
    else:
        raise NameError(f'tokenizer name {args.tokenizer} is not defined.')
    print(f'Tokenizer Model: {args.tokenizer}')

    # Tokenize and convert sentences to indexes
    input_idx_lists = [tokenizer.encode(s, input_max_length) for s in input_sentences]
    target_idx_lists = [tokenizer.encode(s, target_max_length) for s in target_sentences]
    dataset = (input_idx_lists, target_idx_lists)

    # Set seq2seq trainer
    if args.seq2seq == 'rnn':
        trainer = RNNTrainer(vocab_size=tokenizer.vocab_size,
                             batch=args.batch,
                             tokenizer=tokenizer,
                             dataset=dataset,
                             embedding_dim=128,
                             units=256)
    elif args.seq2seq == 'transformer':
        trainer = TransformerTrainer(vocab_size=tokenizer.vocab_size,
                                     batch=args.batch,
                                     tokenizer=tokenizer,
                                     dataset=dataset,
                                     num_layers=4,
                                     d_model=512,
                                     dff=1024,
                                     num_heads=8,
                                     dropout_rate=0.1)
    else:
        raise NameError(f'seq2seq model name {args.seq2seq} is not defined.')
    print(f'Seq2seq Model: {args.seq2seq}')

    # Train epochs
    print(f'Total Epochs: {args.epochs} | Batch Size: {args.batch}')
    for epoch in range(args.epochs):
        print()
        print(f'EPOCH {epoch + 1} / {args.epochs} : Start at {datetime.datetime.now()}')
        start = time.time()
        loss = trainer.train_iter()
        print(f'| Loss: {loss}')
        print(f'| Epoch Time Taken: {time.time() - start} seconds')
        trainer.save()
