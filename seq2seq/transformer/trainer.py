# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import tensorflow as tf
from tqdm import tqdm
from seq2seq.trainer import Trainer
from seq2seq.transformer.model import Transformer
from seq2seq.transformer.utils import CustomSchedule, create_masks


class TransformerTrainer(Trainer):
    def __init__(self,
                 vocab_size,
                 batch,
                 tokenizer,
                 dataset,
                 num_layers,
                 d_model,
                 dff,
                 num_heads,
                 dropout_rate):
        super(TransformerTrainer, self).__init__(name='transformer')
        self.transformer = Transformer(num_layers=num_layers,
                                       d_model=d_model,
                                       num_heads=num_heads,
                                       dff=dff,
                                       input_vocab_size=vocab_size,
                                       target_vocab_size=vocab_size,
                                       pe_input=vocab_size,
                                       pe_target=vocab_size,
                                       rate=dropout_rate)
        self.tokenizer = tokenizer
        self.batch = batch
        self.learning_rate = CustomSchedule(d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        input_data, target_data = dataset
        input_tensors, target_tensors = tf.convert_to_tensor(input_data), tf.convert_to_tensor(target_data)
        self.dataset = tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
        self.dataset = self.dataset.batch(batch, drop_remainder=True)
        self.steps_per_epoch = len(input_tensors) // batch

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.checkpoint_path = f'./seq2seq/transformer/checkpoint/{self.tokenizer.name}/'
        self.checkpoint = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        self.train_info = {
            'seq2seq': self.name,
            'tokenizer': self.tokenizer.name,
            'batch': self.batch,
            'dataset': len(dataset[0]),
            'num_layers': num_layers,
            'd_model': d_model,
            'num_heads': num_heads,
            'dff': dff,
            'input_vocab_size': vocab_size,
            'target_vocab_size': vocab_size,
            'pe_input': vocab_size,
            'pe_target': vocab_size,
            'rate': dropout_rate
        }

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('Transformer Latest Checkpoint Restored.')

    def _loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.tokenizer.PAD_token))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    @tf.function
    def _train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp, self.tokenizer.PAD_token)

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = self._loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

    def train_iter(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        pbar = tqdm(total=self.steps_per_epoch)
        pbar.set_description(f'| {self.name} train iter')
        for (batch, (inp, tar)) in enumerate(self.dataset):
            self._train_step(inp, tar)
            pbar.update(1)
        pbar.close()
        return self.train_loss.result()

    def save(self):
        self.checkpoint_manager.save()

        train_info_file_path = os.path.join(self.checkpoint_path, 'train_info.txt')
        with open(train_info_file_path, 'w') as f:
            json.dump(self.train_info, f, indent=4)
