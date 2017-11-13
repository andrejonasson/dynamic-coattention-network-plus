""" Dynamic Coattention Network Plus, DCN+ [1]

Dynamic Coattention Network (DCN+) consists of an encoder for a (query, document) pair.
The encoder encodes the pair into a single representation in document space. This encoder
implementation can easily be adapted to other use cases than Question Answering.

The decoder implementation takes the encoding and returns answer span logits. Its
application is specific to the SQuAD dataset.

[1] DCN+: Mixed Objective and Deep Residual Coattention for Question Answering, 
    Xiong et al, https://arxiv.org/abs/1711.00106

Shape notation:  
    N = Batch size  
    Q = Query max length  
    D = Document max length  
    H = State size  
    R = Word embedding size  
    ? = Wildcard size  
"""

import tensorflow as tf

def encode(state_size, query, query_length, document, document_length):
    """ DCN+ encoder.
    
    Encodes query document pairs into a document-query representations in document space.

    Args:  
        state_size: A scalar integer. State size of RNN cell encoders.  
        query: A tensor of rank 3, shape [N, Q, R].  
        query_length: A tensor of rank 1, shape [N]. Lengths of queries.  
        document: A tensor of rank 3, shape [N, D, R].  
        document_length: A tensor of rank 1, shape [N]. Lengths of documents.  
    
    Returns:  
        Merged representation of query and document in document space, shape [N, D, 2H].
    """

    def get_cell():
        cell_type = tf.contrib.rnn.LSTMCell
        return cell_type(num_units=state_size)

    with tf.variable_scope('initial_encoder'):
        query_encoding, document_encoding = query_document_encoder(get_cell(), get_cell(), query, query_length, document, document_length)
    
    with tf.variable_scope('coattention_1'):
        summary_q_1, summary_d_1, coattention_d_1 = coattention(query_encoding, query_length, document_encoding, document_length, sentinel=True)
    
    with tf.variable_scope('summary_encoder'):
        summary_q_encoding, summary_d_encoding = query_document_encoder(get_cell(), get_cell(), summary_q_1, query_length, summary_d_1, document_length)
    
    with tf.variable_scope('coattention_2'):
        _, summary_d_2, coattention_d_2 = coattention(summary_q_encoding, query_length, summary_d_encoding, document_length)        

    document_representations = [
        document_encoding,  # E^D_1
        summary_d_encoding, # E^D_2
        summary_d_1,        # S^D_1
        summary_d_2,        # S^D_2
        coattention_d_1,    # C^D_1
        coattention_d_2,    # C^D_2
    ]

    with tf.variable_scope('final_encoder'):
        document_representation = tf.concat(document_representations, 2)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = get_cell(),
            cell_bw = get_cell(),
            dtype = tf.float32,
            inputs = document_representation,
            sequence_length = document_length,
        )
        encoding = tf.concat(outputs, 2)
    return encoding


def query_document_encoder(cell_fw, cell_bw, query, query_length, document, document_length):
    """ DCN+ Query Document Encoder layer.
    
    Forward and backward cells are shared between the bidirectional query and document encoders. 
    The document encoding passes through an additional dense layer with tanh activation.

    Args:  
        cell_fw: RNNCell for forward direction encoding.  
        cell_bw: RNNCell for backward direction encoding.  
        query: A tensor of rank 3, shape [N, Q, ?].  
        query_length: A tensor of rank 1, shape [N]. Lengths of queries.  
        document: A tensor of rank 3, shape [N, D, ?].  
        document_length: A tensor of rank 1, shape [N]. Lengths of documents.  
    Returns:  
        A tuple containing  
            encoding of query, shape [N, Q, 2H].  
            encoding of document, shape [N, D, 2H].
    """
    query_fw_bw_encodings, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = cell_fw,
        cell_bw = cell_bw,
        dtype = tf.float32,
        inputs = query,
        sequence_length = query_length
    )
    query_encoding = tf.concat(query_fw_bw_encodings, 2)        
    
    query_encoding = tf.layers.dense(
        query_encoding, 
        query_encoding.get_shape()[2], 
        activation=tf.tanh,
        kernel_initializer=None
    )

    document_fw_bw_encodings, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = cell_fw,
        cell_bw = cell_bw,
        dtype = tf.float32,
        inputs = document,
        sequence_length = document_length
    )
        
    document_encoding = tf.concat(document_fw_bw_encodings, 2)

    return query_encoding, document_encoding

def concat_sentinel(sentinel_name, other_tensor):
    """ Left concatenates a sentinel vector along `other_tensor`'s second dimension.

    Args:  
        sentinel_name: Variable name of sentinel.  
        other_tensor: A rank 3 Tensor to left concatenate sentinel to.  

    Returns:  
        other_tensor with sentinel.
    """
    sentinel = tf.get_variable(sentinel_name, other_tensor.get_shape()[2], tf.float32)
    sentinel = tf.reshape(sentinel, (1, 1, -1))
    sentinel = tf.tile(sentinel, (tf.shape(other_tensor)[0], 1, 1))
    other_tensor = tf.concat([sentinel, other_tensor], 1)
    return other_tensor

def maybe_mask_affinity(affinity, sequence_length, affinity_mask_value=float('-inf')):
    """ Masks affinity along its third dimension with `affinity_mask_value`.

    Used for masking entries of sequences longer than `sequence_length` prior to 
    applying softmax.  

    Args:  
        affinity: A tensor of rank 3, of shape [N, D or Q, Q or D] where attention logits are in the second dimension.  
        sequence_length: A tensor of rank 1, of shape [N]. Lengths of second dimension of the affinity.  
        affinity_mask_value: (optional) Value to mask affinity with.  
    
    Returns:  
        Masked affinity, same shape as affinity.
    """
    if sequence_length is None:
        return affinity
    score_mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(affinity)[1])
    score_mask = tf.tile(tf.expand_dims(score_mask, 2), (1, 1, tf.shape(affinity)[2]))
    affinity_mask_values = affinity_mask_value * tf.ones_like(affinity)
    return tf.where(score_mask, affinity, affinity_mask_values)


def coattention(query, query_length, document, document_length, sentinel=False):
    """ DCN+ Coattention layer.
    
    Args:  
        query: A tensor of rank 3, shape [N, Q, 2H].  
        query_length: A tensor of rank 1, shape [N]. Lengths of queries without sentinel.  
        document: A tensor of rank 3, shape [N, D, 2H].   
        document_length: A tensor of rank 1, shape [N]. Lengths of documents without sentinel.  
        sentinel: Scalar boolean. If True, then sentinel vectors are temporarily left concatenated 
        to the query's and document's second dimension, letting the attention focus on nothing.  

    Returns:  
        A tuple containing:  
            summary matrix of the query, shape [N, Q, 2H].  
            summary matrix of the document, shape [N, D, 2H].  
            coattention matrix of the document and query in document space, shape [N, D, 2H].
    """

    """
    The symbols in [1] correspond to the following identifiers
        A   = affinity
        A^T = affinity_t
        E^Q = query
        E^D = documenttf.shape(other_tensor)[2]
        S^Q = summary_q
        S^D = summary_d
        C^D = coattention_d
    
    The dimenions' indices in Einstein summation notation are
        n = batch dimension
        q = query dimension
        d = document dimension
        h = hidden state dimension
    """
    if sentinel:
        document = concat_sentinel('document_sentinel', document)
        document_length += 1
        query = concat_sentinel('query_sentinel', query)
        query_length += 1
    # TODO make sure masking is enough
    unmasked_affinity = tf.einsum('ndh,nqh->ndq', document, query)  # [N, D, Q] or [N, 1+D, 1+Q] if sentinel
    affinity = maybe_mask_affinity(unmasked_affinity, document_length)
    attention_p = tf.nn.softmax(affinity, dim=1)
    unmasked_affinity_t = tf.transpose(unmasked_affinity, [0, 2, 1])  # [N, Q, D] or [N, 1+Q, 1+D] if sentinel
    affinity_t = maybe_mask_affinity(unmasked_affinity_t, query_length)
    attention_q = tf.nn.softmax(affinity_t, dim=1)
    summary_q = tf.einsum('ndh,ndq->nqh', document, attention_p)  # [N, Q, 2H] or [N, 1+Q, 2H] if sentinel
    summary_d = tf.einsum('nqh,nqd->ndh', query, attention_q)  # [N, D, 2H] or [N, 1+D, 2H] if sentinel
    if sentinel:
        summary_d = summary_d[:,1:,:]
        summary_q = summary_q[:,1:,:]
        attention_q = attention_q[:,1:,1:]
    coattention_d = tf.einsum('nqh,nqd->ndh', summary_q, attention_q)
    return summary_q, summary_d, coattention_d


def decode(encoding):
    """ Decodes encoding to answer span logits.

    Args:  
        encoding: Document representation, shape [N, D, ?].  
    
    Returns:  
        A tuple containing  
            Logit for answer span start position, shape [N, D].  
            Logit for answer span end position, shape [N, D].
    """
    
    with tf.variable_scope('decode_start'):
        start_logit = tf.layers.dense(encoding, 1)
        start_logit = tf.squeeze(start_logit)
    
    # TODO condition decode_end on decode_start
    with tf.variable_scope('decode_end'):
        end_logit = tf.layers.dense(encoding, 1)
        end_logit = tf.squeeze(end_logit)

    return start_logit, end_logit
