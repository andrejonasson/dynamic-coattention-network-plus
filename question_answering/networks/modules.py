import tensorflow as tf
from tensorflow.python.framework import function

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


def maybe_dropout(keep_prob, is_training=False):
    return tf.cond(tf.convert_to_tensor(is_training), lambda: keep_prob, lambda: 1.0)


def max_product_span(start, end, length):
    """ Finds answer span with the largest answer span probability product
    
    Dynamic programming approach for finding maximum product in linear time is applied
    to efficiently find the solution.

    Args:  
        start: Tensor of shape [N, D]. Probabilities for start of span.  
        end: Tensor of shape [N, D]. Probabilities for end of span.  
        length: Tensor of shape [N]. Length of each document.  
    
    Returns:  
        Tuple containing two tensors of shape [N] with start and end indices 
        for spans with maximum probability product

    TODO: implement max span length
    """
    batch_size = tf.shape(start)[0]
    i = tf.zeros((batch_size,), dtype=tf.int32)
    j = tf.zeros((batch_size,), dtype=tf.int32)
    span_start = tf.zeros((batch_size,), dtype=tf.int32)
    span_end = tf.zeros((batch_size,), dtype=tf.int32)
    argmax_start = tf.zeros((batch_size,), dtype=tf.int32)
    max_product = tf.zeros((batch_size,), dtype=tf.float32)

    loop_vars = [i, j, span_start, span_end, argmax_start, max_product]

    def cond(i, j, span_start, span_end, argmax_start, max_product): 
        return tf.reduce_any(tf.less(j, length))
    
    def body(i, j, span_start, span_end, argmax_start, max_product):
        i = tf.where(tf.less(j, length), j, i)
        
        # get current largest start probability up to i, compare with 
        # new possible start probability, update if necessary
        start_prob = tf.gather_nd(start, tf.stack([tf.range(batch_size), i], axis=1))
        max_start_prob = tf.gather_nd(start, tf.stack([tf.range(batch_size), argmax_start], axis=1))
        argmax_start = tf.where(start_prob > max_start_prob, i, argmax_start)
        max_start_prob = tf.where(start_prob > max_start_prob, start_prob, max_start_prob)

        # calculate new product, if new product is greater update span and max product
        end_prob = tf.gather_nd(end, tf.stack([tf.range(batch_size), i], axis=1))
        new_product = max_start_prob * end_prob
        span_start = tf.where(new_product > max_product, argmax_start, span_start)
        span_end = tf.where(new_product > max_product, i, span_end)
        max_product = tf.where(new_product > max_product, new_product, max_product)
        return i, j+1, span_start, span_end, argmax_start, max_product

    i, j, span_start, span_end, argmax_start, max_product = tf.while_loop(cond, body, loop_vars)
    return span_start, span_end


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

    with tf.variable_scope('decode_end'):
        end_logit = tf.layers.dense(encoding, 1)
        end_logit = tf.squeeze(end_logit)

    return start_logit, end_logit


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """Identity operation whose gradient is converted to a `Tensor`.
    Currently, the gradient to `tf.concat` is particularly expensive to
    compute if dy is an `IndexedSlices` (a lack of GPU implementation
    forces the gradient operation onto CPU).  This situation occurs when
    the output of the `tf.concat` is eventually passed to `tf.gather`.
    It is sometimes faster to convert the gradient to a `Tensor`, so as
    to get the cheaper gradient for `tf.concat`.  To do this, replace
    `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.
    Args:
    x: A `Tensor`.
    Returns:
    The input `Tensor`.
    """
    return x


def cell_factory(cell_type, state_size, is_training, input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0):
    if cell_type.lower() == 'gru':
        cell = tf.contrib.rnn.GRUCell(num_units=state_size)
    elif cell_type.lower() == 'lstm':
        cell = tf.contrib.rnn.LSTMCell(num_units=state_size)
    input_keep_prob = maybe_dropout(input_keep_prob, is_training)
    output_keep_prob = maybe_dropout(output_keep_prob, is_training)
    state_keep_prob = maybe_dropout(state_keep_prob, is_training)
    dropout_cell = tf.contrib.rnn.DropoutWrapper(
        cell, 
        input_keep_prob=input_keep_prob, 
        output_keep_prob=output_keep_prob, 
        state_keep_prob=state_keep_prob
    )
    return dropout_cell
