import tensorflow as tf

def maybe_dropout(keep_prob, is_training=False):
    return tf.cond(tf.convert_to_tensor(is_training), lambda: keep_prob, lambda: 1.0)