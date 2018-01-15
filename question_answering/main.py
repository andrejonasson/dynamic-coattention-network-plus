import os
import json
import logging
import itertools
from datetime import datetime
from timeit import default_timer as timer
from os.path import join as pjoin

import tensorflow as tf
import numpy as np

from preprocessing.squad_preprocess import tokenize
from utils import initialize_vocab, get_normalized_train_dir, f1, get_data_paths, exact_match
from preprocessing.qa_data import UNK_ID, PAD_ID
from networks.dcn_model import DCN
from dataset import SquadDataset, pad_sequence

logging.basicConfig(level=logging.INFO)

# Mode
tf.app.flags.DEFINE_string('mode', 'train', 'Mode to use, train/eval/shell/overfit')

# Training hyperparameters
tf.app.flags.DEFINE_integer("max_steps", 50000, "Steps until training loop stops.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

tf.app.flags.DEFINE_boolean("exponential_decay", False, "Whether to use exponential decay.")
tf.app.flags.DEFINE_float("decay_steps", 4000, "Number of steps for learning rate to decay by decay_rate")
tf.app.flags.DEFINE_boolean("staircase", True, "Whether staircase decay (use of integer division in decay).")
tf.app.flags.DEFINE_float("decay_rate", 0.75, "Learning rate.")

tf.app.flags.DEFINE_boolean("clip_gradients", True, "Whether to clip gradients.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")


# Model hyperparameters
tf.app.flags.DEFINE_string("model", 'dcnplus', "Model to train or evaluate, dcnplus / baseline")
tf.app.flags.DEFINE_string("cell", 'lstm', "Cell type to use for RNN, 'gru'/'lstm'.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("trainable_initial_state", False, "Make RNNCell initial states trainable.")  # Not implemented
tf.app.flags.DEFINE_integer("trainable_embeddings", False, "Make embeddings trainable.")
tf.app.flags.DEFINE_float("input_keep_prob", 0.8, "Encoder: Fraction of units randomly kept of inputs to RNN.")
tf.app.flags.DEFINE_float("output_keep_prob", 1.0, "Encoder: Fraction of units randomly kept of outputs from RNN.")
tf.app.flags.DEFINE_float("state_keep_prob", 1.0, "Encoder: Fraction of units randomly kept of encoder states in RNN.")
tf.app.flags.DEFINE_float("encoding_keep_prob", 1.0, "Encoder: Fraction of encoding output kept.")
tf.app.flags.DEFINE_float("final_input_keep_prob", 0.7, "Encoder: Fraction of units randomly kept of inputs to final encoder RNN.")

# DCN+ hyperparameters
tf.app.flags.DEFINE_integer("pool_size", 4, "Number of units the maxout network pools.")
tf.app.flags.DEFINE_integer("max_iter", 4, "Maximum number of iterations of decoder.")
tf.app.flags.DEFINE_float("keep_prob", 0.80, "Decoder: Fraction of units randomly kept on non-recurrent connections.")

# Character embeddings
tf.app.flags.DEFINE_integer("char_vocab_size", 4, "Number of characters in vocabulary.")  # TODO will be overridden by loaded vocab, probably not needed
tf.app.flags.DEFINE_integer("char_embedding_size", 8, "Size of character embeddings.")
tf.app.flags.DEFINE_integer("max_word_length", 15, "Maximum number of characters per word.")

# Data hyperparameters
tf.app.flags.DEFINE_integer("use_char_cnn", False, "Whether to use character embeddings to build word vectors.")
tf.app.flags.DEFINE_integer("max_question_length", 25, "Maximum question length.")
tf.app.flags.DEFINE_integer("max_paragraph_length", 400, "Maximum paragraph length and the output size of your model.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")

# Evaluation arguments
tf.app.flags.DEFINE_integer("eval_batches", 80, "Number of batches of size batch_size to use for evaluation.")

# Directories etc.
tf.app.flags.DEFINE_string("model_name", datetime.now().strftime('%y%m%d_%H%M%S'), "Models name, used for folder management.")
tf.app.flags.DEFINE_string("data_dir", os.path.join("..", "data", "squad"), "SQuAD directory (default ../data/squad)") 
tf.app.flags.DEFINE_string("train_dir", os.path.join("..", "checkpoints"), "Training directory to save the model parameters (default: ../checkpoints).")
tf.app.flags.DEFINE_integer("print_every", 50, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("vocab_path", os.path.join("..", "data", "squad", "vocab.dat"), "Path to vocab file (default: ../data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ../data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS

# TODO hyperparameter random search
# TODO implement early stopping
# TODO Write final Dev set eval to a file that's easily inspected
# TODO performance by question and paragraph length (later by )

def reverse_indices(indices, rev_vocab):
    """ Recovers words from embedding indices

    Args:  
        indices: Integer indices to recover words for.  
        rev_vocab: Reverse vocabulary. Dictionary mapping indices to words.  
    
    Returns:  
        String of words with space as separation
    """
    return ' '.join([rev_vocab[idx] for idx in indices if idx != PAD_ID])


def do_shell(model, dev):
    """ Interactive shell

    Type a question, write next for the next paragraph or enter a blank for another human's question.  

    Args:  
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders  
        question, question_length, paragraph, paragraph_length  
        dev: Development set
    """
    # what is is_training if import_meta_graph
    checkpoint_dir = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    # TODO no logs
    saver = tf.train.Saver()
    with tf.Session() as session:
        if False:  # load_meta
            last_meta = next(reversed([f for f in os.listdir(checkpoint_dir) if '.meta' in f]))
            saver = tf.train.import_meta_graph(os.path.join(last_meta))
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        print('HINT: Input as question "next" for next paragraph')
        while True:
            original_question, paragraphs, question_lengths, paragraph_lengths, answers = dev.get_batch(1)
            for i in itertools.count():
                paragraph = reverse_indices(paragraphs[0], rev_vocab)
                if not i:
                    print('\n')
                    print(paragraph, end='\n\n')
                
                question_input = input('QUESTION: ')

                if question_input == 'next':
                    break
                elif question_input:
                    question = [vocab.get(word, UNK_ID) for word in tokenize(question_input)]
                    question, question_length = pad_sequence(question, FLAGS.max_question_length)
                    questions, question_lengths = [question], [question_length]
                else:
                    question_words = reverse_indices(original_question[0], rev_vocab)
                    questions = original_question
                    print(question_words)
                
                feed_dict = model.fill_feed_dict(questions, paragraphs, question_lengths, paragraph_lengths)
                
                if False: #load_meta
                    start, end = session.run(['prediction/answer_start:0', 'prediction/answer_end:0'], feed_dict)
                    start, end = start[0], end[0]
                else:
                    start, end = session.run(model.answer, feed_dict)
                    start, end = start[0], end[0]

                answer_idxs = paragraphs[0][start:end+1]
                answer_words = ''.join(reverse_indices(answer_idxs, rev_vocab))
                print(f'COMPUTER: {answer_words}')

                if not question_input:
                    start, end = answers[0]
                    correct_answer_idxs = paragraphs[0][start:end+1]
                    correct_answer = ''.join(reverse_indices(correct_answer_idxs, rev_vocab))
                    print(f'HUMAN: {correct_answer}')
                print()


def parameter_space_size():
    """ Parameter space size information """
    num_parameters = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
    logging.info('Number of parameters %d' % num_parameters)
    for v in tf.trainable_variables():
        logging.info(f'Variable {v} has {v.get_shape().num_elements()} entries')


def do_eval(model, train, dev):
    """ Evaluates a model on training and development set

    Args:  
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders  
        question, question_length, paragraph, paragraph_length  
        train: Training set  
        dev: Development set
    """
    checkpoint_dir = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    parameter_space_size()
    saver = tf.train.Saver()
    # TODO add loop to run over all checkpoints in folder, 
    # Training session
    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        print('Evaluation in progress.', flush=True)

        # Train/Dev Evaluation
        start_evaluate = timer()
        
        prediction, truth = multibatch_prediction_truth(session, model, train, FLAGS.eval_batches)
        train_f1 = f1(prediction, truth)
        train_em = exact_match(prediction, truth)

        prediction, truth = multibatch_prediction_truth(session, model, dev, FLAGS.eval_batches)
        dev_f1 = f1(prediction, truth)
        dev_em = exact_match(prediction, truth)

        logging.info(f'Train/Dev F1: {train_f1:.3f}/{dev_f1:.3f}')
        logging.info(f'Train/Dev EM: {train_em:.3f}/{dev_em:.3f}')
        logging.info(f'Time to evaluate: {timer() - start_evaluate:.1f} sec')


def multibatch_prediction_truth(session, model, data, num_batches=None, random=False):
    """ Returns batches of predictions and ground truth answers.

    Args:  
        session: TensorFlow Session.  
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders.  
        question, question_length, paragraph, paragraph_length.  
        data: SquadDataset data to do minibatch evaluation on.  
        num_batches: Number of batches of size FLAGS.batch_size to evaluate over. `None` for whole data set.  
        random: True for random and possibly overlapping batches. False for deterministic sequential non-overlapping batches.  
    
    Returns:  
        Tuple of  
            Predictions, tuple of two numpy arrays containing start and end of answer spans  
            Truth, list of tuples containing start and end of answer spans
    """
    if num_batches is None:
        num_batches = data.length // FLAGS.batch_size
    truth = []
    start = []
    end = []
    for i in range(num_batches):
        if random:
            q, p, ql, pl, a = data.get_batch(FLAGS.batch_size, replace=False)
        else:
            begin_idx = i * FLAGS.batch_size
            q, p, ql, pl, a = data[begin_idx:begin_idx+FLAGS.batch_size]
        answer_start, answer_end = session.run(model.answer, model.fill_feed_dict(q, p, ql, pl))
        start.append(answer_start)
        end.append(answer_end)
        truth.extend(a)
    start = np.concatenate(start)
    end = np.concatenate(end)
    prediction = (start, end)
    return prediction, truth


def do_train(model, train):
    """ Trains a model

    Args:  
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders  
        question, question_length, paragraph, paragraph_length  
        train: Training set
    """
    checkpoint_dir = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    
    hooks = [
        tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
        tf.train.NanTensorHook(model.loss)
    ]
    parameter_space_size()

    losses = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Training session  
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           checkpoint_dir=checkpoint_dir, 
                                           save_summaries_steps=20,
                                           config=config) as session:
        while not session.should_stop():
            feed_dict = model.fill_feed_dict(*train.get_batch(FLAGS.batch_size))
            fetch_dict = {
                'step': tf.train.get_global_step(),
                'loss': model.loss,
                'train': model.train
            }
            result = session.run(fetch_dict, feed_dict)
            step = result['step']
            losses.append(result['loss'])
            
            # Moving Average loss
            if step == 1 or step == 10 or step == 100 or step % FLAGS.print_every == 0:
                mean_loss = sum(losses)/len(losses)
                losses = []
                print(f'Step {step}, loss {mean_loss:.2f}')


def save_flags():
    """ Saves flags in checkpoints folder without overwriting previous """
    model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    for i in itertools.count():
        json_path = os.path.join(FLAGS.train_dir, FLAGS.model_name, f"flags_{i}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                if json.load(f) == FLAGS.__flags:
                    break
        else:
            with open(json_path, 'w') as f:
                json.dump(FLAGS.__flags, f, indent=4)
            break


def test_overfit(model, train):
    """ Tests that model can overfit on small datasets.

    Args:  
        model: QA model that has an instance variable 'answer' that returns answer span and takes placeholders  
        question, question_length, paragraph, paragraph_length  
        train: Training set  
    """
    epochs = 100
    test_size = 32
    steps_per_epoch = 10
    train.question, train.paragraph, train.question_length, train.paragraph_length, train.answer = train[:test_size]
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_start = timer()
            for step in range(steps_per_epoch):
                feed_dict = model.fill_feed_dict(*train[:test_size])
                fetch_dict = {
                    'step': tf.train.get_global_step(),
                    'loss': model.loss,
                    'train': model.train
                }
                result = session.run(fetch_dict, feed_dict)
                loss = result['loss']

                if (step == 0 and epoch == 0):
                    print(f'Entropy - Result: {loss:.2f}, Expected (approx.): {2*np.log(FLAGS.max_paragraph_length):.2f}')
                if step == steps_per_epoch-1:
                    print(f'Cross entropy: {loss:.2f}')
                    train.length = test_size
                    prediction, truth = multibatch_prediction_truth(session, model, train, 1)
                    overfit_f1 = f1(prediction, truth)
                    print(f'F1: {overfit_f1:.2f}')
            global_step = tf.train.get_global_step().eval()
            print(f'Epoch took {timer() - epoch_start:.2f} s (step: {global_step})')


def main(_):
    """ Typical usage

    For <model_name> see your folder name in ../checkpoints. 

    Training
    ``` sh
    $ python main.py --mode train --model <model> (if restoring or naming a model: --model_name <model_name>)
    ```
    
    Evaluation
    ``` sh
    $ python main.py --mode eval --model <model> --model_name <model_name>
    ```

    Shell
    ``` sh
    $ python main.py --mode shell --model <model> --model_name <model_name>
    ```
    """
    # Load data
    train = SquadDataset(*get_data_paths(FLAGS.data_dir, name='train'), 
                         max_question_length=FLAGS.max_question_length, 
                         max_paragraph_length=FLAGS.max_paragraph_length)
    dev = SquadDataset(*get_data_paths(FLAGS.data_dir, name='val'), 
                         max_question_length=FLAGS.max_question_length, 
                         max_paragraph_length=FLAGS.max_paragraph_length) # change to eval to zero if too long

    # logging.info(f'Train/Dev size {train.length}/{dev.length}')

    # Load embeddings
    embed_path = FLAGS.embed_path or pjoin(FLAGS.data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = np.load(embed_path)['glove']  # 115373
    
    is_training = (FLAGS.mode == 'train' or FLAGS.mode == 'overfit')
    
    # Build model
    if FLAGS.model in ('baseline', 'mixed', 'dcnplus'):
        model = DCN(embeddings, FLAGS.__flags, is_training=is_training)
    elif FLAGS.model == 'cat':
        from networks.cat import Graph
        model = Graph(embeddings, is_training=is_training)
    else:
        raise ValueError(f'{FLAGS.model} is not a supported model')
    
    # Run mode
    if FLAGS.mode == 'train':
        save_flags()
        do_train(model, train)
    elif FLAGS.mode == 'eval':
        do_eval(model, train, dev)
    elif FLAGS.mode == 'overfit':
        test_overfit(model, train)
    elif FLAGS.mode == 'shell':
        do_shell(model, dev)
    else:
        raise ValueError(f'Incorrect mode entered, {FLAGS.mode}')


if __name__ == "__main__":
    tf.app.run()

