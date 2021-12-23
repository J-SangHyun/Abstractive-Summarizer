# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import tensorflow as tf
from tqdm import tqdm
from seq2seq.module import Module
from seq2seq.rnn.model import Encoder, Decoder


class RNNModule(Module):
    def __init__(self, tokenizer):
        super(RNNModule, self).__init__(name='rnn')
        self.tokenizer = tokenizer
        self.checkpoint_path = f'./seq2seq/rnn/checkpoint/{self.tokenizer.name}/'
        train_info_file_path = os.path.join(self.checkpoint_path, 'train_info.txt')

        with open(train_info_file_path, 'r') as f:
            train_info = json.load(f)

        self.encoder = Encoder(vocab_size=train_info['vocab_size'],
                               embedding_dim=train_info['embedding_dim'],
                               enc_units=train_info['units'],
                               batch_sz=train_info['batch'])
        self.decoder = Decoder(vocab_size=train_info['vocab_size'],
                               embedding_dim=train_info['embedding_dim'],
                               dec_units=train_info['units'],
                               batch_sz=train_info['batch'])
        self.units = train_info['units']
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print('RNN Latest Checkpoint Restored.')

    def predict(self, indexes, max_length):
        inputs = tf.expand_dims(indexes, 0)
        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.tokenizer.CLS_token], 0)
        result = [self.tokenizer.CLS_token]

        for i in range(max_length):
            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            dec_input = tf.expand_dims([predicted_id], 0)
            result.append(int(predicted_id))

            if int(predicted_id) == self.tokenizer.SEP_token:
                break
        return result

    def predict_batch(self, indexes_list, max_length, batch):
        result = []
        input_tensors = tf.convert_to_tensor(indexes_list)
        input_dataset = tf.data.Dataset.from_tensor_slices(input_tensors)
        input_dataset = input_dataset.batch(batch, drop_remainder=False)

        steps_per_epoch = len(indexes_list) // batch + (0 if len(indexes_list) % batch == 0 else 1)
        pbar = tqdm(total=steps_per_epoch)
        pbar.set_description(f'{self.name} batch prediction...')
        for (batch, inp) in enumerate(input_dataset):
            batch_size = inp.shape[0]
            hidden = [tf.zeros((batch_size, self.units))]
            enc_out, enc_hidden = self.encoder(inp, hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tokenizer.CLS_token] * batch_size, 1)
            output = [[self.tokenizer.CLS_token] * batch_size]
            output = tf.convert_to_tensor(output)
            output = tf.reshape(output, shape=(batch_size, -1))

            for i in range(max_length):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_out)
                predicted_id = tf.expand_dims(tf.cast(tf.argmax(predictions, axis=-1), tf.int32), 1)
                dec_input = predicted_id
                output = tf.concat([output, predicted_id], axis=-1)

            for i in range(output.shape[0]):
                result.append([int(j) for j in output[i]])
            pbar.update(1)
        pbar.close()
        return result
