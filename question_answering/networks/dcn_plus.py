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
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score

from networks.modules import maybe_mask_affinity, convert_gradient_to_tensor


def baseline_encode(cell_factory, final_cell_factory, query, query_length, document, document_length, keep_prob=1.0):
    """ Baseline Encoder that encodes questions and paragraphs into one representation.  

    It first encodes the question and paragraphs using a shared BiLSTM, then uses a 
    one layer coattention similar to Dynamic Coattention Network's [1]. Finally, it concatenates 
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
        initial = cell_factory()
        query_encoding, document_encoding = query_document_encoder(initial, query, query_length, document, document_length, bidirectional=False)
        #query_encoding = tf.nn.dropout(query_encoding, keep_prob)
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
        document_representation = tf.concat(document_representations, 2)
        final = final_cell_factory()
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = final,
            cell_bw = final,
            dtype = tf.float32,
            inputs = document_representation,
            sequence_length = document_length,
        )
        encoding = tf.concat(outputs, 2)
    return encoding  # N x P x 2H


def dcn_encode(cell_factory, final_cell_factory, query, query_length, document, document_length, keep_prob=1.0):
    """ DCN Encoder that encodes questions and paragraphs into one representation.  

    It first encodes the question and paragraphs using a shared LSTM, then uses a 
    one layer coattention as in Dynamic Coattention Network's [1]. Finally, concatenates 
    the initial encoding, summary of question for document and the coattention to build 
    a final encoding using a separate BiLSTM.  

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
        initial = cell_factory()
        query_encoding, document_encoding = query_document_encoder(initial, query, query_length, document, document_length, bidirectional=False)
        #query_encoding = tf.nn.dropout(query_encoding, keep_prob) # test if wanted
        query_encoding = tf.layers.dense(
            query_encoding, 
            query_encoding.get_shape()[2], 
            activation=tf.tanh,
            #kernel_initializer=tf.initializers.identity()  # Not mentioned in paper, the assumption is that identity transform is closer to optimal than a noise matrix
        )
    
    with tf.variable_scope('coattention'):
        _, summary_d, coattention_d = coattention(query_encoding, query_length, document_encoding, document_length, sentinel=True)
    
    document_representations = [document_encoding, summary_d, coattention_d]

    with tf.variable_scope('final_encoder'):
        document_representation = tf.concat(document_representations, 2)
        final = final_cell_factory()
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = final,
            cell_bw = final,
            dtype = tf.float32,
            inputs = document_representation,
            sequence_length = document_length,
        )
        encoding = tf.concat(outputs, 2)
    return encoding  # N x P x 2H


def dcnplus_encode(cell_factory, final_cell_factory, query, query_length, document, document_length, keep_prob=1.0):
    """ DCN+ deep residual coattention encoder.
    
    Encodes query document pairs into a document-query representations in document space.

    Args:  
        cell_factory: Function of zero arguments returning an RNNCell.  
        final_cell_factory: Function of zero arguments returning an RNNCell. Applied in final encoder layer.  
        query: Tensor of rank 3, shape [N, Q, R].  
        query_length: Tensor of rank 1, shape [N]. Lengths of queries.  
        document: Tensor of rank 3, shape [N, D, R].  
        document_length: Tensor of rank 1, shape [N]. Lengths of documents.  
    
    Returns:  
        Merged representation of query and document in document space, shape [N, D, 2H].
    """

    with tf.variable_scope('initial_encoder'):
        initial = cell_factory()
        query_encoding, document_encoding = query_document_encoder(initial, query, query_length, document, document_length)
        #query_encoding = tf.nn.dropout(query_encoding, keep_prob)
        query_encoding = tf.layers.dense(
            query_encoding, 
            query_encoding.get_shape()[2], 
            activation=tf.tanh,
            #kernel_initializer=tf.initializers.identity()  # Not mentioned in paper, the assumption is that identity transform is closer to optimal than a noise matrix
        )
    
    with tf.variable_scope('coattention_1'):
        summary_q_1, summary_d_1, coattention_d_1 = coattention(query_encoding, query_length, document_encoding, document_length, sentinel=True)
    
    with tf.variable_scope('summary_encoder'):
        summary = cell_factory()
        summary_q_encoding, summary_d_encoding = query_document_encoder(summary, summary_q_1, query_length, summary_d_1, document_length)
    
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
        document_representation = convert_gradient_to_tensor(tf.concat(document_representations, 2))
        final = final_cell_factory()
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = final,
            cell_bw = final,
            dtype = tf.float32,
            inputs = document_representation,
            sequence_length = document_length,
        )
        encoding = convert_gradient_to_tensor(tf.concat(outputs, 2))
    return encoding


def query_document_encoder(cell, query, query_length, document, document_length, bidirectional=True):
    """ DCN+ Query Document Encoder layer.
    
    Forward and backward cells are shared between the bidirectional query and document encoders.  

    Args:  
        cell: RNNCell for encoding.   
        query: Tensor of rank 3, shape [N, Q, ?].  
        query_length: Tensor of rank 1, shape [N]. Lengths of queries.  
        document: Tensor of rank 3, shape [N, D, ?].  
        document_length: Tensor of rank 1, shape [N]. Lengths of documents.  

    Returns:  
        A tuple containing  
            encoding of query, shape [N, Q, 2H].  
            encoding of document, shape [N, D, 2H].
    """
    
    if bidirectional:
        query_fw_bw_encodings, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell,
            cell_bw = cell,
            dtype = tf.float32,
            inputs = query,
            sequence_length = query_length
        )
        query_encoding = convert_gradient_to_tensor(tf.concat(query_fw_bw_encodings, 2))

        document_fw_bw_encodings, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = cell,
            cell_bw = cell,
            dtype = tf.float32,
            inputs = document,
            sequence_length = document_length
        )    
        document_encoding = convert_gradient_to_tensor(tf.concat(document_fw_bw_encodings, 2))
    else:
        query_encoding, _ = tf.nn.dynamic_rnn(
            cell = cell,
            dtype = tf.float32,
            inputs = query,
            sequence_length = query_length
        )

        document_encoding, _ = tf.nn.dynamic_rnn(
            cell = cell,
            dtype = tf.float32,
            inputs = document,
            sequence_length = document_length
        )
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
    other_tensor = convert_gradient_to_tensor(tf.concat([sentinel, other_tensor], 1))
    return other_tensor


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
    
    The dimensions' indices in Einstein summation notation are
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
    encoding_start = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), start], axis=1))  # May be causing UserWarning
    encoding_end = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), end], axis=1))
    return convert_gradient_to_tensor(tf.concat([encoding_start, encoding_end], axis=1))


def dcn_decode(encoding, document_length, state_size=100, pool_size=4, max_iter=4, keep_prob=1.0):
    """ DCN+ Dynamic Decoder.

    Builds decoder graph that iterates over possible solutions to problem
    until it returns same answer in two consecutive iterations or reaches `max_iter` iterations.  

    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        state_size: Scalar integer. Size of state and highway network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
        max_iter: Scalar integer. Maximum number of attempts for answer span start and end to settle.  
        keep_prob: Scalar float. Probability of keeping units during dropout.
    Returns:  
        A tuple containing  
            TensorArray of answer span logits for each iteration.  
            TensorArray of logit masks for each iteration.
    """

    with tf.variable_scope('decoder_loop', reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(encoding)[0]
        lstm_dec = tf.contrib.rnn.LSTMCell(num_units=state_size)
        lstm_dec = tf.contrib.rnn.DropoutWrapper(lstm_dec, input_keep_prob=keep_prob)

        # initialise loop variables
        start = tf.zeros((batch_size,), dtype=tf.int32)
        end = document_length - 1
        answer = tf.stack([start, end], axis=1)
        state = lstm_dec.zero_state(batch_size, dtype=tf.float32)
        not_settled = tf.tile([True], (batch_size,))
        logits = tf.TensorArray(tf.float32, size=max_iter, clear_after_read=False)

        def calculate_not_settled_logits(not_settled, answer, output, prev_logit):
            enc_masked = tf.boolean_mask(encoding, not_settled)
            output_masked = tf.boolean_mask(output, not_settled)
            answer_masked = tf.boolean_mask(answer, not_settled)
            document_length_masked = tf.boolean_mask(document_length, not_settled)
            new_logit = decoder_body(enc_masked, output_masked, answer_masked, state_size, pool_size, document_length_masked, keep_prob)
            new_idx = tf.boolean_mask(tf.range(batch_size), not_settled)
            logit = tf.dynamic_stitch([tf.range(batch_size), new_idx], [prev_logit, new_logit])  # TODO test that correct
            return logit

        for i in range(max_iter):
            if i > 1:
                tf.summary.scalar(f'not_settled_iter_{i+1}', tf.reduce_sum(tf.cast(not_settled, tf.float32)))
            
            output, state = lstm_dec(start_and_end_encoding(encoding, answer), state)
            if i == 0:
                logit = decoder_body(encoding, output, answer, state_size, pool_size, document_length, keep_prob)
            else:
                prev_logit = logits.read(i-1)
                logit = tf.cond(
                    tf.reduce_any(not_settled),
                    lambda: calculate_not_settled_logits(not_settled, answer, output, prev_logit),
                    lambda: prev_logit
                )
            start_logit, end_logit = logit[:, :, 0], logit[:, :, 1]
            start = tf.argmax(start_logit, axis=1, output_type=tf.int32)
            end = tf.argmax(end_logit, axis=1, output_type=tf.int32)
            new_answer = tf.stack([start, end], axis=1)
            if i == 0:
                not_settled = tf.tile([True], (batch_size,))
            else:
                not_settled = tf.reduce_any(tf.not_equal(answer, new_answer), axis=1)
            not_settled = tf.reshape(not_settled, (batch_size,))  # needed to establish dimensions
            answer = new_answer
            logits = logits.write(i, logit)

    return logits


def dcn_decode_simplified(encoding, document_length, state_size=100, pool_size=4, max_iter=4, keep_prob=1.0):
    with tf.variable_scope('decoder_loop', reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(encoding)[0]
        lstm_dec = tf.contrib.rnn.LSTMCell(num_units=state_size)
        lstm_dec = tf.contrib.rnn.DropoutWrapper(lstm_dec, input_keep_prob=keep_prob)

        # initialise loop variables
        start = tf.zeros((batch_size,), dtype=tf.int32)
        end = document_length - 1
        answer = tf.stack([start, end], axis=1)
        state = lstm_dec.zero_state(batch_size, dtype=tf.float32)
        logits = tf.TensorArray(tf.float32, size=max_iter, clear_after_read=False)

        for i in range(max_iter):
            output, state = lstm_dec(start_and_end_encoding(encoding, answer), state)
            logit = decoder_body(encoding, output, answer, state_size, pool_size, document_length, keep_prob)
            start_logit, end_logit = logit[:, :, 0], logit[:, :, 1]
            start = tf.argmax(start_logit, axis=1, output_type=tf.int32)
            end = tf.argmax(end_logit, axis=1, output_type=tf.int32)
            answer = tf.stack([start, end], axis=1)
            logits = logits.write(i, logit)
        
    return logits

def decoder_body(encoding, state, answer, state_size, pool_size, document_length, keep_prob=1.0):
    """ Decoder feedforward network.  

    Calculates answer span start and end logits.  

    Args:  
        encoding: Tensor of rank 3, shape [N, D, xH]. Query-document encoding.  
        state: Tensor of rank 2, shape [N, D, C]. Current state of decoder state machine.  
        answer: Tensor of rank 2, shape [N, 2]. Current iteration's answer.  
        state_size: Scalar integer. Hidden units of highway maxout network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout network.  
        keep_prob: Scalar float. Input dropout keep probability for maxout layers.
    
    Returns:  
        Tensor of rank 3, shape [N, D, 2]. Answer span logits for answer start and end.
    """
    maxlen = tf.shape(encoding)[1]
    
    def highway_maxout_network(answer):
        span_encoding = start_and_end_encoding(encoding, answer)
        r_input = convert_gradient_to_tensor(tf.concat([state, span_encoding], axis=1))
        r_input = tf.nn.dropout(r_input, keep_prob)
        r = tf.layers.dense(r_input, state_size, use_bias=False, activation=tf.tanh)
        r = tf.expand_dims(r, 1)
        r = tf.tile(r, (1, maxlen, 1))
        highway_input = convert_gradient_to_tensor(tf.concat([encoding, r], 2))
        logit = highway_maxout(highway_input, state_size, pool_size, keep_prob)
        #alpha = two_layer_mlp(highway_input, state_size, keep_prob=keep_prob)
        logit = _maybe_mask_score(logit, document_length, -1e30)
        return logit

    with tf.variable_scope('start'):
        alpha = highway_maxout_network(answer)

    with tf.variable_scope('end'):
        updated_start = tf.argmax(alpha, axis=1, output_type=tf.int32)
        updated_answer = tf.stack([updated_start, answer[:, 1]], axis=1)
        beta = highway_maxout_network(updated_answer)
    
    return tf.stack([alpha, beta], axis=2)


def two_layer_mlp(inputs, hidden_size, keep_prob=1.0):
    """ Two layer MLP network.

    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to network.  
        hidden_size: Scalar integer. Hidden units of network.  
        keep_prob: Scalar float. Input dropout keep probability.  

    Returns:  
        Tensor of rank 2, shape [N, D]. Logits.
    """
    
    inputs = tf.nn.dropout(inputs, keep_prob)
    layer1 = tf.layers.dense(inputs, hidden_size, activation=tf.nn.relu)
    output = tf.layers.dense(layer1, 1)
    output = tf.squeeze(output, -1)
    return output


def highway_maxout(inputs, hidden_size, pool_size, keep_prob=1.0):
    """ Highway maxout network.

    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to network.  
        hidden_size: Scalar integer. Hidden units of highway maxout network.  
        pool_size: Scalar integer. Number of units that are max pooled in maxout layer.  
        keep_prob: Scalar float. Input dropout keep probability for maxout layers.  

    Returns:  
        Tensor of rank 2, shape [N, D]. Logits.
    """
    layer1 = maxout_layer(inputs, hidden_size, pool_size, keep_prob)
    layer2 = maxout_layer(layer1, hidden_size, pool_size, keep_prob)
    
    highway = convert_gradient_to_tensor(tf.concat([layer1, layer2], -1))
    output = maxout_layer(highway, 1, pool_size, keep_prob)
    output = tf.squeeze(output, -1)
    return output


def mixture_of_experts():
    pass


def maxout_layer(inputs, outputs, pool_size, keep_prob=1.0):
    """ Maxout layer

    Args:  
        inputs: Tensor of rank 3, shape [N, D, ?]. Inputs to layer.  
        outputs: Scalar integer, number of outputs.  
        pool_size: Scalar integer, number of units to max pool over.  
        keep_prob: Scalar float, input dropout keep probability.  
    
    Returns:  
        Tensor, shape [N, D, outputs]. Result of maxout layer.
    """
    
    inputs = tf.nn.dropout(inputs, keep_prob)
    pool = tf.layers.dense(inputs, outputs*pool_size)
    pool = tf.reshape(pool, (-1, tf.shape(inputs)[1], outputs, pool_size))
    output = tf.reduce_max(pool, -1)
    return output


def dcn_loss(logits, answer_span, max_iter):
    """ Calulates cumulative loss over the iterations

    Args:  
        logits: TensorArray of Tensors of rank 3 [N, D, 2] of size max_iter. Contains logits of start 
        and end of answer span  
        answer_span: Integer placeholder containing indices of true answer spans [N, 2].
        max_iter: Scalar integer, Maximum number of iterations the decoder is run.

    Returns:  
        Mean cross entropy loss across iterations and batch. Mean is used instead of sum to make loss be 
        on same scale as other more traditional methods.
    """
    batch_size = tf.shape(answer_span)[0]
    logits = convert_gradient_to_tensor(logits.concat())
    answer_span_repeated = tf.tile(answer_span, (max_iter, 1))

    start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:, :, 0], labels=answer_span_repeated[:, 0], name='start_loss')
    end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:, :, 1], labels=answer_span_repeated[:, 1], name='end_loss')
    start_loss = tf.stack(tf.split(start_loss, max_iter), axis=1)
    end_loss = tf.stack(tf.split(end_loss, max_iter), axis=1)

    loss_per_example = tf.reduce_mean(start_loss + end_loss, axis=1)
    loss = tf.reduce_mean(loss_per_example)
    return loss
