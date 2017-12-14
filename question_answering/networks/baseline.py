import tensorflow as tf
from networks.modules import maybe_mask_affinity


def encode(cell_factory, question, question_length, paragraph, paragraph_length):
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
        cell_factory: Function of zero arguments returning an RNNCell.
        question: A tensor of rank 3, shape [N, Q, R]. Word embeddings for each word in the question.
        question_length: A tensor of rank 1, shape [N]. Lengths of questions.
        paragraph: A tensor of rank 3, shape [N, P, R]. Word embeddings for each word in the paragraphs.
        paragraph_length: A tensor of rank 1, shape [N]. Lengths of paragraphs.
    
    Returns:
        Rank 3 tensor with shape [N, P, 2H].
        
    """

    with tf.variable_scope('initial_encoder'):
        # Shared RNN for question and paragraph encoding
        cell_fw = cell_factory()
        cell_bw = cell_factory()
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
            cell_fw = cell_factory(),
            cell_bw = cell_factory(),
            dtype = tf.float32,
            inputs = paragraph_with_coattention,
            sequence_length = paragraph_length,
        )
        encoding = tf.concat(outputs, 2)
    return encoding  # N x P x 2H
