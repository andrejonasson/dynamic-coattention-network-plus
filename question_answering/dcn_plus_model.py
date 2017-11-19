import time
import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score
from tensorflow import variable_scope
from dcn_plus import encode, decode, loss
from evaluate import exact_match_score, f1_score


# TODO output from decoder + loss definition (_maybe_mask_score?)

class DCNPlus:
    def __init__(self, pretrained_embeddings, hparams):
        self.hparams = copy.copy(hparams)
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
            encoding = encode(self.hparams['state_size'], q_embeddings, self.question_length, p_embeddings, self.paragraph_length)
            logits = decode(encoding, self.hparams['state_size'], self.hparams['pool_size'], self.hparams['max_iter'])

        with tf.variable_scope('loss'):
            self.loss = loss(logits, self.answer_span)

        with tf.variable_scope('last_iter_results')
            last_iter_logit = logits.read(hparams['max_iter']-1)
            start_logit, end_logit = last_iter_logit[:,:,0], last_iter_logit[:,:,1]
            start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logit, labels=self.answer_span[:, 0], name='start_loss')
            end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logit, labels=self.answer_span[:, 1], name='end_loss')
            loss_per_example = start_loss + end_loss
            last_loss = tf.reduce_mean(loss_per_example)

            self.answer = (tf.argmax(start_logit, axis=1), tf.argmax(end_logit, axis=1))

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
        tf.summary.scalar('cross_entropy_last_iter', last_loss)
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
    