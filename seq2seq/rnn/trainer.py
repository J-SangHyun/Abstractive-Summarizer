# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import tensorflow as tf
from tqdm import tqdm
from seq2seq.trainer import Trainer
from seq2seq.rnn.model import Encoder, Decoder


class RNNTrainer(Trainer):
    def __init__(self,
                 vocab_size,
                 batch,
                 tokenizer,
                 dataset,
                 embedding_dim,
                 units):
        super(RNNTrainer, self).__init__(name='rnn')

        # Network architecture
        self.encoder = Encoder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               enc_units=units,
                               batch_sz=batch)
        self.decoder = Decoder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               dec_units=units,
                               batch_sz=batch)

        self.batch = batch
        self.tokenizer = tokenizer

        input_data, target_data = dataset
        input_tensors, target_tensors = tf.convert_to_tensor(input_data), tf.convert_to_tensor(target_data)
        self.dataset = tf.data.Dataset.from_tensor_slices((input_tensors, target_tensors))
        self.dataset = self.dataset.batch(batch, drop_remainder=True)
        self.steps_per_epoch = len(input_tensors) // batch

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.checkpoint_path = f'./seq2seq/rnn/checkpoint/{self.tokenizer.name}/'
        self.checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=5)

        self.train_info = {
            'seq2seq': self.name,
            'tokenizer': self.tokenizer.name,
            'batch': self.batch,
            'dataset': len(dataset[0]),
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'units': units
        }

        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print('RNN Latest Checkpoint Restored.')

    def _loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, self.tokenizer.PAD_token))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def _train_step(self, inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.tokenizer.CLS_token] * self.batch, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self._loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train_iter(self):
        enc_hidden = self.encoder.initialize_hidden_state()
        total_loss = 0

        pbar = tqdm(total=self.steps_per_epoch)
        pbar.set_description(f'| {self.name} train iter')
        for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
            batch_loss = self._train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            pbar.update(1)
        pbar.close()
        return total_loss

    def save(self):
        self.checkpoint_manager.save()

        train_info_file_path = os.path.join(self.checkpoint_path, 'train_info.txt')
        with open(train_info_file_path, 'w') as f:
            json.dump(self.train_info, f, indent=4)
