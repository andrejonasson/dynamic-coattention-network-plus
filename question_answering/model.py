import time
import copy
import numpy as np

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score
from tensorflow import variable_scope

from evaluate import exact_match_score, f1_score


# TODO output from decoder + loss definition (_maybe_mask_score?)

class QASystem:
    def __init__(self, encoder, decoder, pretrained_embeddings, hparams):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.hparams = copy.copy(hparams)
        self.encode = encoder
        self.decode = decoder
        self.pretrained_embeddings = pretrained_embeddings

        # Setup placeholders
        self.question = tf.placeholder(tf.int32, (None, None), name='question')
        self.question_length = tf.placeholder(tf.int32, (None,), name='question_length')
        self.paragraph = tf.placeholder(tf.int32, (None, None), name='paragraph')
        self.paragraph_length = tf.placeholder(tf.int32, (None,), name='paragraph_length')
        self.answer_span = tf.placeholder(tf.int32, (None, 2), name='answer_span')
        self.keep_prob = tf.placeholder(tf.float32, (), name='keep_prob')

        with tf.variable_scope('embeddings'):
            embedded_vocab = tf.Variable(self.pretrained_embeddings, name='shared_embedding', trainable=self.hparams['trainable_embeddings'], dtype=tf.float32)  
            q_embeddings = tf.nn.embedding_lookup(embedded_vocab, self.question)
            p_embeddings = tf.nn.embedding_lookup(embedded_vocab, self.paragraph)
        
        with tf.variable_scope('prediction'):
            encoding = self.encode(self.hparams['state_size'], q_embeddings, self.question_length, p_embeddings, self.paragraph_length)
            self.start_logit, self.end_logit = self.decode(encoding)

            # naive answer - need to search for max of a_s*a_e (dynamic programming)
            self.answer = (tf.argmax(self.start_logit, axis=1), tf.argmax(self.end_logit, axis=1))
            # TODO _maybe_mask_score  [tf.nn.softmax(self.start_logit), tf.nn.softmax(self.end_logit)]

        with tf.variable_scope('loss'):
            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_logit, labels=self.answer_span[:, 0], name='start_loss')
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_logit, labels=self.answer_span[:, 1], name='end_loss')
            loss_per_example = start_loss + end_loss
            self.loss = tf.reduce_mean(loss_per_example)

        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('train'):
            if self.hparams['exponential_decay']:
                lr = tf.train.exponential_decay(learning_rate=self.hparams['learning_rate'], 
                                                global_step=global_step, 
                                                decay_steps=self.hparams['decay_steps'], 
                                                decay_rate=self.hparams['decay_rate'], 
                                                staircase=self.hparams['staircase']) 
            else:
                lr = self.hparams['learning_rate']
            optimizer = tf.train.AdamOptimizer(lr)
            grad, tvars = zip(*optimizer.compute_gradients(self.loss))
            if self.hparams['clip_gradients']:
                grad, _ = tf.clip_by_global_norm(grad, self.hparams['max_gradient_norm'], name='gradient_clipper')  
            grad_norm = tf.global_norm(grad)
            self.train = optimizer.apply_gradients(zip(grad, tvars), global_step=global_step, name='apply_grads')
        
        tf.summary.scalar('cross_entropy', self.loss)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('grad_norm', grad_norm)

    def fill_feed_dict(self, question, paragraph, question_length, paragraph_length, answer_span=None, keep_prob=1.0):
        feed_dict = {
            self.question: question,
            self.paragraph: paragraph,
            self.question_length: question_length, 
            self.paragraph_length: paragraph_length,
            self.keep_prob: keep_prob  # need scheme for feeding in dropout properly
        }

        # TODO Why does it require answer_span placeholder when answer_span is not on the path to model.answer?
        if answer_span is not None:
            feed_dict[self.answer_span] = answer_span

        return feed_dict
    