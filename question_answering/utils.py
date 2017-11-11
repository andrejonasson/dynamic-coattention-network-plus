import os
from os.path import join as pjoin
from collections import Counter
import tensorflow as tf


def initialize_vocab(vocab_path):
    # CS224n
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def get_normalized_train_dir(train_dir):
    # CS224n
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def get_data_paths(data_dir, name='train'):
    question_file = pjoin(data_dir, f'{name}.ids.question')
    paragraph_file = pjoin(data_dir, f'{name}.ids.context')
    answer_file = pjoin(data_dir, f'{name}.span')
    return question_file, paragraph_file, answer_file

def evaluate(session, model, data, size=100):
    q, p, ql, pl, a = data.get_batch(size, replace=False)
    # TODO Why does it require answer_span placeholder when answer_span is not on the path to model.answer?
    a_pred = session.run(model.answer, model.fill_feed_dict(q, p, ql, pl, a)) 
    total = 0
    f1_total = 0

    for i, truth in enumerate(a):
        total += 1
        prediction = a_pred[0][i], a_pred[1][i]
        f1 = f1_score(prediction, truth)
        f1_total += f1
    f1_total /= total

    return f1_total

def f1_score(prediction, truth):
    start, end = truth
    true_range = range(start, end+1)
    start_pred, end_pred = prediction
    prediction_range = range(start_pred, end_pred+1)
    common = Counter(prediction_range) & Counter(true_range)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_range)
    recall = 1.0 * num_same / len(true_range)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1