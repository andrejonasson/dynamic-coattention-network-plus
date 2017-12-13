import tensorflow as tf


def maybe_dropout(keep_prob, is_training=False):
    return tf.cond(tf.convert_to_tensor(is_training), lambda: keep_prob, lambda: 1.0)


def max_span_product(start, end, length):
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
