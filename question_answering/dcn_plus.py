""" Dynamic Coattention Network Plus, DCN+ [1]

Dynamic Coattention Network (DCN+) consists of an encoder for a (query, document) pair and
a dynamic decoder that iteratively searches for an answer span (answer_start, answer_end)
that answers the query.

The encoder encodes the pair into a single representation in document space. This encoder
implementation can easily be adapted to other use cases than SQuAD dataset.

The decoder implementation is passed the encoding and returns answer span logits. Its
application is specific to the SQuAD dataset.

[1] DCN+: Mixed Objective and Deep Residual Coattention for Question Answering,
    Xiong et al, https://arxiv.org/abs/1711.00106

Shape notation:  
    N = Batch size  
    Q = Query max length  
    D = Document max length  
    H = State size  
    R = Word embedding size  
    xH = x times the state size H
    C = Decoder state size
    ? = Wildcard size  
"""

import tensorflow as tf

def encode(state_size, query, query_length, document, document_length):
    """ DCN+ deep residual coattention encoder.
    
    Encodes query document pairs into a document-query representations in document space.

    Args:  
        state_size: Scalar integer. State size of RNN cell encoders.  
        query: Tensor of rank 3, shape [N, Q, R].  
        query_length: Tensor of rank 1, shape [N]. Lengths of queries.  
        document: Tensor of rank 3, shape [N, D, R].  
        document_length: Tensor of rank 1, shape [N]. Lengths of documents.  
    
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
        query: Tensor of rank 3, shape [N, Q, ?].  
        query_length: Tensor of rank 1, shape [N]. Lengths of queries.  
        document: Tensor of rank 3, shape [N, D, ?].  
        document_length: Tensor of rank 1, shape [N]. Lengths of documents.  

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
        other_tensor: Tensor of rank 3 to left concatenate sentinel to.  

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
        affinity: Tensor of rank 3, shape [N, D or Q, Q or D] where attention logits are in the second dimension.  
        sequence_length: Tensor of rank 1, shape [N]. Lengths of second dimension of the affinity.  
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
        query: Tensor of rank 3, shape [N, Q, 2H].  
        query_length: Tensor of rank 1, shape [N]. Lengths of queries without sentinel.  
        document: Tensor of rank 3, shape [N, D, 2H].   
        document_length: Tensor of rank 1, shape [N]. Lengths of documents without sentinel.  
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
        E^D = document
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


def start_and_end_encoding(encoding, answer):
    """ Gathers the encodings representing the start and end of the answer span passed
    and concatenates the encodings.

    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        answer: Tensor of rank 2. Answer span.  
    
    Returns:
        Tensor of rank 2 [N, 2xH], containing the encodings of the start and end of the answer span
    """
    batch_size = tf.shape(encoding)[0]
    start, end = answer[:, 0], answer[:, 1]
    encoding_start = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), start], axis=1))
    encoding_end = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), end], axis=1))
    return tf.concat([encoding_start, encoding_end], axis=1)


def decode(encoding, state_size=100, pool_size=4, max_iter=4):
    """ DCN+ Dynamic Decoder.

    Dynamically builds decoder graph. 

    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        state_size: Scalar integer. Size of state and highway network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
        max_iter: Scalar integer. Maximum number of attempts for answer span start and end to settle.  
    
    Returns:  
        A tuple containing  
            TensorArray of answer span logits for each iteration.  
            TensorArray of logit masks for each iteration.
    """

    with tf.variable_scope('decoder_loop', reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(encoding)[0]  # N
        maxlen = tf.shape(encoding)[1]
        lstm_dec = tf.contrib.rnn.LSTMCell(num_units=state_size)

        def cond(i, state, not_settled, answer, logits, logit_masks):
            return tf.less(i, max_iter) & tf.reduce_any(not_settled)  # check if gives computational advantage over just dynamic stitch, o.w. remove

        def loop_body(i, state, not_settled, answer, logits, logit_masks):
            output, state = lstm_dec(start_and_end_encoding(encoding, answer), state)
            
            def calculate_not_settled_logits():
                enc_masked = tf.boolean_mask(encoding, not_settled)
                output_masked = tf.boolean_mask(output, not_settled)
                answer_masked = tf.boolean_mask(answer, not_settled)
                new_logit = decoder_body(enc_masked, output_masked, answer_masked, state_size, pool_size)
                new_idx = tf.boolean_mask(tf.range(batch_size), not_settled)
                logit = logits.read(i-1)
                logit = tf.dynamic_stitch([tf.range(batch_size), new_idx], [logit, new_logit])  # TODO test that correct  # TODO consumes previous value ?
                return logit

            logit = tf.cond(
                tf.equal(i, 0) | tf.reduce_all(not_settled),
                lambda: decoder_body(encoding, output, answer, state_size, pool_size),
                calculate_not_settled_logits,
            )
            start = tf.argmax(logit[:, :, 0], axis=1, output_type=tf.int32)
            end = tf.argmax(logit[:, :, 1], axis=1, output_type=tf.int32)
            new_answer = tf.stack([start, end], axis=1)
            not_settled = tf.cond(
                tf.equal(i, 0), 
                lambda: tf.tile([True], [batch_size]),
                lambda: tf.reduce_any(tf.not_equal(answer, new_answer), axis=1)
            )
            logit_masks = logit_masks.write(i, not_settled)
            logits = logits.write(i, logit)
            return i + 1, state, not_settled, new_answer, logits, logit_masks
        
        # initialise loop variables
        # TODO possibly just choose first and last encoding
        start = tf.random_uniform((batch_size,), maxval=maxlen, dtype=tf.int32)
        end = tf.random_uniform((batch_size,), minval=tf.reduce_max(start), maxval=maxlen, dtype=tf.int32)
        answer = tf.stack([start, end], axis=1)
        state = lstm_dec.zero_state(batch_size, dtype=tf.float32)
        not_settled = tf.tile([True], (batch_size,))
        
        logits = tf.TensorArray(tf.float32, size=max_iter)
        logit_masks = tf.TensorArray(tf.float32, size=max_iter)
        loop_vars = [0, state, not_settled, answer, logits, logit_masks]
        i, _, _, answer, logits, logit_masks = tf.while_loop(cond, loop_body, loop_vars)
        
        (max_iter - i) * [logits.read(i-1)]

    # tf.summary.scalar(tf.reduce_mean(tf.cast(logits_masks.read(i-1), tf.float32)))
    # TODO add a summary "mean_i" to see the mean number of iterations
    alphabeta = logits.read(i-1)
    return alphabeta[:,:,0], alphabeta[:,:,1]# alpha, beta # answer_span_logits


def decoder_body(encoding, state, answer, state_size, pool_size):
    """ Decoder feedforward network.  

    Calculates answer span start and end logits.  

    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        state: Tensor of rank 2, shape [N, D, C]. Current state of decoder state machine.  
        answer: Tensor of rank 2, shape [N, 2]. Current iteration's answer.  
        state_size: Scalar integer. Hidden units of highway maxout network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
    
    Returns:  
        Tensor of rank 3, shape [N, D, 2]. Answer span logits for answer start and end.
    """
    maxlen = tf.shape(encoding)[1]
    span_encoding = start_and_end_encoding(encoding, answer)

    with tf.variable_scope('start'):  # TODO need reuse, although currently getting it from tf.AUTO_REUSE
        r_input = tf.concat([state, span_encoding], axis=1)
        r = tf.layers.dense(r_input, state_size, use_bias=False, activation=tf.tanh)  # outputs  # [N, ]
        r = tf.expand_dims(r, 1)
        r = tf.tile(r, (1, maxlen, 1))
        alpha = highway_maxout(tf.concat([encoding, r], 2), state_size, pool_size)
    
    with tf.variable_scope('end'):
        r_input = tf.concat([state, span_encoding], axis=1)
        r = tf.layers.dense(r_input, state_size, use_bias=False, activation=tf.tanh)
        r = tf.expand_dims(r, 1)
        r = tf.tile(r, (1, maxlen, 1))
        beta = highway_maxout(tf.concat([encoding, r], 2), state_size, pool_size)
    
    return tf.stack([alpha, beta], axis=2)
    

def highway_maxout(inputs, hidden_size, pool_size):
    """ Highway maxout network.

    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to network.  
        hidden_size: Scalar integer. Hidden units of highway maxout network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
    
    Returns:  
        Tensor of rank 2, shape [N, D]. Logits.
    """
    layer1 = maxout_layer(inputs, hidden_size, pool_size)
    layer2 = maxout_layer(layer1, hidden_size, pool_size)
    
    highway = tf.concat([layer1, layer2], -1)
    output = maxout_layer(highway, 1, pool_size)
    return tf.reshape(output, (tf.shape(inputs)[0], tf.shape(inputs)[1]))  # TODO temp


def mixture_of_experts():
    pass


def maxout_layer(inputs, outputs, pool_size):
    pool = tf.layers.dense(inputs, outputs*pool_size)
    pool = tf.reshape(pool, (-1, outputs, pool_size))  # possibly skip via num_units=outputs in next line
    output = tf.contrib.layers.maxout(pool, 1)
    output = tf.squeeze(output, -1)
    return output


def loss(logits: tf.TensorArray, iteration_mask:tf.TensorArray=None, mask_settled=False):
    batch_size, maxlen, _ = logits.get(0).shape()
    max_iter = logits.size()

    logits = logits.concat()
    logits.reshape(batch_size, max_iter, -1)
    return loss


def naive_decode(encoding):
    """ Decodes encoding to answer span logits.

    Args:  
        encoding: Document representation, shape [N, D, xH].  
    
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
