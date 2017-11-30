import tensorflow as tf

def maybe_mask_affinity(affinity, sequence_length, affinity_mask_value=float('-inf')):
    """ Masks affinity along its third dimension with `affinity_mask_value`
    Useful for masking entries for sequences longer than sequence_length prior to 
    applying softmax.
    """
    if sequence_length is None:
        return affinity
    score_mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(affinity)[1])
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), (1, 1, tf.shape(affinity)[2]))
    affinity_mask_values = affinity_mask_value * tf.ones_like(affinity)
    return tf.where(score_mask, affinity, affinity_mask_values)

def encode(state_size, question, question_length, paragraph, paragraph_length):
    """ Baseline Encoder that encodes questions and paragraphs into one representation.

    It first encodes the question and paragraphs using a shared BiLSTM, then uses a 
    one layer coattention similar to Dynamic Coattention Network's [1]. Finally, concatenates 
    the initial encoding and the coattention to build a final encoding using
    a separate BiLSTM.

    [1] Dynamic Coattention Networks For Question Answering, Xiong et al, 
        https://arxiv.org/abs/1611.01604

    N = Batch size
    P = Paragraph max length
    Q = Question max length
    H = state_size
    R = Word embedding

    Args:
        state_size: A scalar integer. Number of units for RNN encoding.
        question: A tensor of rank 3, shape [N, Q, R]. Word embeddings for each word in the question.
        question_length: A tensor of rank 1, shape [N]. Lengths of questions.
        paragraph: A tensor of rank 3, shape [N, P, R]. Word embeddings for each word in the paragraphs.
        paragraph_length: A tensor of rank 1, shape [N]. Lengths of paragraphs.
    
    Returns:
        Rank 3 tensor with shape [N, P, 2H].
        
    """

    def get_cell():
        cell_type = tf.contrib.rnn.LSTMCell
        return cell_type(num_units=state_size)

    with tf.variable_scope('initial_encoder'):
        # Shared RNN for question and paragraph encoding
        cell_fw = get_cell()
        cell_bw = get_cell()
        q_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell_fw,
            cell_bw = cell_bw,
            dtype = tf.float32,
            inputs = question,
            sequence_length = question_length
        )
        question_encoding = tf.concat(q_outputs, 2)

        p_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell_fw,
            cell_bw = cell_bw,
            dtype = tf.float32,
            inputs = paragraph,
            sequence_length = paragraph_length
        )
        paragraph_encoding = tf.concat(p_outputs, 2)
    
    with tf.variable_scope('coattention'):  # make sure masking is enough
        unmasked_affinity = tf.einsum('nph,nqh->npq', paragraph_encoding, question_encoding)  # N x P x Q
        affinity = maybe_mask_affinity(unmasked_affinity, paragraph_length)
        attention_p = tf.nn.softmax(affinity, dim=1)  # N x P x Q
        affinity_T = maybe_mask_affinity(tf.transpose(unmasked_affinity, [0, 2, 1]), question_length)
        attention_q = tf.nn.softmax(affinity_T, dim=1)  # N x Q x P
        summary_q = tf.einsum('nph,npq->nhq', paragraph_encoding, attention_p)  # N x 2H x Q
        coattention_d = tf.einsum('nhq,nqp->nph', summary_q, attention_q) # N x P x 2H
    
    with tf.variable_scope('final_encoder'):
        paragraph_with_coattention = tf.concat([paragraph_encoding, coattention_d], 2)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = get_cell(),
            cell_bw = get_cell(),
            dtype = tf.float32,
            inputs = paragraph_with_coattention,
            sequence_length = paragraph_length,
        )
        encoding = tf.concat(outputs, 2)
    return encoding  # N x P x 2H

def decode(encoding):
    """ Decodes encoding to logits used to find answer span

    Args:
        encoding: Document representation, shape [N, D, ?].
    
    Returns:
        A tuple containing
            Logit for answer span start position, shape [N, D]
            Logit for answer span end position, shape [N, D]
    """
    
    with tf.variable_scope('decode_start'):
        start_logit = tf.layers.dense(encoding, 1)
        start_logit = tf.squeeze(start_logit, axis=-1)
    
    # TODO condition decode_end on decode_start
    with tf.variable_scope('decode_end'):
        end_logit = tf.layers.dense(encoding, 1)
        end_logit = tf.squeeze(end_logit, axis=-1)

    return start_logit, end_logit
