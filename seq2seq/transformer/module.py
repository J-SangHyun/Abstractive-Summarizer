# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import tensorflow as tf
from tqdm import tqdm
from seq2seq.module import Module
from seq2seq.transformer.model import Transformer
from seq2seq.transformer.utils import CustomSchedule, create_masks


class TransformerModule(Module):
    def __init__(self, tokenizer):
        super(TransformerModule, self).__init__(name='transformer')
        self.tokenizer = tokenizer
        self.checkpoint_path = f'./seq2seq/transformer/checkpoint/{self.tokenizer.name}/'
        train_info_file_path = os.path.join(self.checkpoint_path, 'train_info.txt')

        with open(train_info_file_path, 'r') as f:
            train_info = json.load(f)

        self.transformer = Transformer(num_layers=train_info['num_layers'],
                                       d_model=train_info['d_model'],
                                       num_heads=train_info['num_heads'],
                                       dff=train_info['dff'],
                                       input_vocab_size=train_info['input_vocab_size'],
                                       target_vocab_size=train_info['target_vocab_size'],
                                       pe_input=train_info['pe_input'],
                                       pe_target=train_info['pe_target'],
                                       rate=train_info['rate'])
        self.learning_rate = CustomSchedule(train_info['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.checkpoint = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print('Transformer Latest Checkpoint Restored.')

    def predict(self, indexes, max_length):
        encoder_input = tf.expand_dims(indexes, 0)
        decoder_input = [self.tokenizer.CLS_token]
        output = tf.expand_dims(decoder_input, 0)
        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input,
                                                                             output,
                                                                             self.tokenizer.PAD_token)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            output = tf.concat([output, predicted_id], axis=-1)

            if int(predicted_id) == self.tokenizer.SEP_token:
                break
        result = tf.squeeze(output, axis=0)
        return [int(i) for i in result]

    def predict_batch(self, indexes_list, max_length, batch):
        result = []
        input_tensors = tf.convert_to_tensor(indexes_list)
        input_dataset = tf.data.Dataset.from_tensor_slices(input_tensors)
        input_dataset = input_dataset.batch(batch, drop_remainder=False)

        steps_per_epoch = len(indexes_list)//batch + (0 if len(indexes_list) % batch == 0 else 1)
        pbar = tqdm(total=steps_per_epoch)
        pbar.set_description(f'{self.name} batch prediction...')
        for (batch, inp) in enumerate(input_dataset):
            batch_size = inp.shape[0]
            output = [[self.tokenizer.CLS_token] * batch_size]
            output = tf.convert_to_tensor(output)
            output = tf.reshape(output, shape=(batch_size, -1))
            for i in range(max_length):
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                                 output,
                                                                                 self.tokenizer.PAD_token)
                predictions, _ = self.transformer(inp, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
                predictions = predictions[:, -1:, :]
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                output = tf.concat([output, predicted_id], axis=-1)

            for i in range(output.shape[0]):
                result.append([int(j) for j in output[i]])
            pbar.update(1)
        pbar.close()
        return result
