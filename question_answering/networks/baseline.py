import tensorflow as tf
from networks.modules import maybe_mask_affinity
from networks.dcn_plus import query_document_encoder, coattention

def encode(cell_factory, final_cell_factory, query, query_length, document, document_length):
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
        final_cell_factory: Function of zero arguments returning an RNNCell. Applied in final encoder layer.  
        query: A tensor of rank 3, shape [N, Q, R]. Word embeddings for each word in the question.  
        query_length: A tensor of rank 1, shape [N]. Lengths of questions.  
        document: A tensor of rank 3, shape [N, P, R]. Word embeddings for each word in the paragraphs.  
        document_length: A tensor of rank 1, shape [N]. Lengths of paragraphs.  
    
    Returns:  
        Rank 3 tensor with shape [N, P, 2H].
    """

    with tf.variable_scope('initial_encoder'):
        query_encoding, document_encoding = query_document_encoder(cell_factory(), cell_factory(), query, query_length, document, document_length)
        query_encoding = tf.layers.dense(
            query_encoding, 
            query_encoding.get_shape()[2], 
            activation=tf.tanh,
            #kernel_initializer=tf.initializers.identity()  # Not mentioned in paper, the assumption is that identity transform is closer to optimal than a noise matrix
        )
    
    with tf.variable_scope('coattention'):
        _, _, coattention_d = coattention(query_encoding, query_length, document_encoding, document_length)
    
    document_representations = [document_encoding, coattention_d]

    with tf.variable_scope('final_encoder'):
        paragraph_with_coattention = tf.concat(document_representations, 2)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = final_cell_factory(),
            cell_bw = final_cell_factory(),
            dtype = tf.float32,
            inputs = paragraph_with_coattention,
            sequence_length = document_length,
        )
        encoding = tf.concat(outputs, 2)
    return encoding  # N x P x 2H
