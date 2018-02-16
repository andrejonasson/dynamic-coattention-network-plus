import copy

import tensorflow as tf

from networks.modules import maybe_dropout, max_product_span, naive_decode, cell_factory, char_cnn_word_vectors, _maybe_mask_to_start
from networks.dcn_plus import baseline_encode, dcn_encode, dcnplus_encode, dcn_decode, dcn_loss
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score

class DCN:
    """ Builds graph for DCN-type models  
    
    Baseline model - DCN-like Encoder with naive decoder  
    Mixed model - DCN+ Encoder with naive decoder  
    DCN - DCN Encoder with DCN decoder  
    DCN+ - DCN+ Encoder with DCN decoder  

    Args:  
        pretrained_embeddings: Pretrained embeddings.  
        hparams: dictionary of all hyperparameters for models.  
    """
    def __init__(self, pretrained_embeddings, hparams):
        self.hparams = copy.copy(hparams)
        self.pretrained_embeddings = pretrained_embeddings

        # Setup placeholders
        self.question = tf.placeholder(tf.int32, (None, None), name='question')
        self.question_length = tf.placeholder(tf.int32, (None,), name='question_length')
        self.paragraph = tf.placeholder(tf.int32, (None, None), name='paragraph')
        self.paragraph_length = tf.placeholder(tf.int32, (None,), name='paragraph_length')
        self.answer_span = tf.placeholder(tf.int32, (None, 2), name='answer_span')
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')   # replace with tf.placeholder_with_default

        # Word embeddings
        with tf.variable_scope('embeddings'):
            embedded_vocab = tf.Variable(self.pretrained_embeddings, name='shared_embedding', trainable=hparams['trainable_embeddings'], dtype=tf.float32)  
            q_embeddings = tf.nn.embedding_lookup(embedded_vocab, self.question)
            p_embeddings = tf.nn.embedding_lookup(embedded_vocab, self.paragraph)
        
        # Character embeddings to word vectors
        if hparams['use_char_cnn']:
            self.question_chars = tf.placeholder(tf.int32, (None, None, self.hparams['max_word_length']), name='question_chars')
            self.paragraph_chars = tf.placeholder(tf.int32, (None, None, self.hparams['max_word_length']), name='paragraph_chars')
        
            with tf.variable_scope('char_cnn', reuse=tf.AUTO_REUSE):
                filter_widths = [5]  # TODO add as comma separated FLAGS argument
                num_filters = [100]  # TODO add as comma separated FLAGS argument
                char_embeddings = tf.get_variable('char_embeddings', shape=[self.hparams['char_vocab_size'], self.hparams['char_embedding_size']], dtype=tf.float32)
                q_word_vectors = char_cnn_word_vectors(self.question_chars, char_embeddings, filter_widths, num_filters)
                p_word_vectors = char_cnn_word_vectors(self.paragraph_chars, char_embeddings, filter_widths, num_filters)  # reusing filters
                q_embeddings = tf.concat([q_embeddings, q_word_vectors], axis=2)
                p_embeddings = tf.concat([p_embeddings, p_word_vectors], axis=2)

        # Setup RNN Cells
        cell = lambda: cell_factory(hparams['cell'], hparams['state_size'], self.is_training, hparams['input_keep_prob'], hparams['output_keep_prob'], hparams['state_keep_prob'])
        final_cell = lambda: cell_factory(hparams['cell'], hparams['state_size'], self.is_training, hparams['final_input_keep_prob'], hparams['output_keep_prob'], hparams['state_keep_prob'])  # TODO TEMP

        # Setup Encoders
        with tf.variable_scope('prediction'):
            if hparams['model'] == 'baseline':
                self.encode = baseline_encode
            elif hparams['model'] == 'dcn':
                self.encode = dcn_encode
            else:
                self.encode = dcnplus_encode
            encoding = self.encode(cell, final_cell, q_embeddings, self.question_length, p_embeddings, self.paragraph_length, keep_prob=maybe_dropout(hparams['keep_prob'], self.is_training))
            encoding = tf.nn.dropout(encoding, keep_prob=maybe_dropout(hparams['encoding_keep_prob'], self.is_training))
        
        # Decoder, loss and prediction mechanism are different for baseline/mixed and dcn/dcn_plus
        if hparams['model'] in ('baseline', 'mixed'):
            with tf.variable_scope('prediction'):
                start_logit, end_logit = naive_decode(encoding, hparams['state_size'], self.paragraph_length)
                start_prob, end_prob = tf.nn.softmax(start_logit), tf.nn.softmax(end_logit)
                self.answer = max_product_span(start_prob, end_prob, self.paragraph_length)

            with tf.variable_scope('loss'):
                start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logit, labels=self.answer_span[:, 0], name='start_loss')
                end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logit, labels=self.answer_span[:, 1], name='end_loss')
                loss_per_example = start_loss + end_loss
                self.loss = tf.reduce_mean(loss_per_example)

        elif hparams['model'] in ('dcn', 'dcnplus'):
            with tf.variable_scope('prediction'):
                logits = dcn_decode(encoding, self.paragraph_length, hparams['state_size'], hparams['pool_size'], hparams['max_iter'], keep_prob=maybe_dropout(hparams['keep_prob'], self.is_training))
                last_iter_logit = logits.read(hparams['max_iter']-1)
                start_logit, end_logit = last_iter_logit[:,:,0], last_iter_logit[:,:,1]
                start = tf.argmax(start_logit, axis=1, name='answer_start')
                if hparams['force_end_gt_start']:
                    end_logit = _maybe_mask_to_start(end_logit, start, -1e30)
                if hparams['max_answer_length'] > 0:
                    end_logit = _maybe_mask_score(end_logit, start+hparams['max_answer_length'], -1e30)
                self.answer = (start, tf.argmax(end_logit, axis=1, name='answer_end'))

            with tf.variable_scope('loss'):
                self.loss = dcn_loss(logits, self.answer_span, max_iter=hparams['max_iter'])

            # Solely for diagnostics purposes
            with tf.variable_scope('last_iter_loss'):
                start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logit, labels=self.answer_span[:, 0], name='start_loss')
                end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logit, labels=self.answer_span[:, 1], name='end_loss')
                last_loss = tf.reduce_mean(start_loss + end_loss)
            tf.summary.scalar('cross_entropy_last_iter', last_loss)

        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('train'):
            if hparams['exponential_decay']:
                lr = tf.train.exponential_decay(learning_rate=hparams['learning_rate'], 
                                                global_step=global_step, 
                                                decay_steps=hparams['decay_steps'], 
                                                decay_rate=hparams['decay_rate'], 
                                                staircase=hparams['staircase']) 
            else:
                lr = hparams['learning_rate']
            optimizer = tf.train.AdamOptimizer(lr)
            grad, tvars = zip(*optimizer.compute_gradients(self.loss))
            if hparams['clip_gradients']:
                grad, _ = tf.clip_by_global_norm(grad, hparams['max_gradient_norm'], name='gradient_clipper')  
            grad_norm = tf.global_norm(grad)
            self.train = optimizer.apply_gradients(zip(grad, tvars), global_step=global_step, name='apply_grads')
        
        tf.summary.scalar('cross_entropy', self.loss)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('grad_norm', grad_norm)
    

    def fill_feed_dict(self, question, paragraph, question_length, paragraph_length, answer_span=None, is_training=False):
        feed_dict = {
            self.question: question,
            self.paragraph: paragraph,
            self.question_length: question_length, 
            self.paragraph_length: paragraph_length,
            self.is_training: is_training
        }

        if answer_span is not None:
            feed_dict[self.answer_span] = answer_span

        return feed_dict