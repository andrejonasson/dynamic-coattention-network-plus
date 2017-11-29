import os
import json
import logging
from datetime import datetime
from timeit import default_timer as timer
from os.path import join as pjoin
from collections import Counter

import tensorflow as tf
import numpy as np

from utils import initialize_vocab, get_normalized_train_dir, evaluate, get_data_paths
from cat import Graph
from baseline_model import Baseline
from dcn_plus_model import DCNPlus
from dataset import SquadDataset

logging.basicConfig(level=logging.INFO)

# Mode
tf.app.flags.DEFINE_string('mode', 'train', 'Mode to use, train or eval')

# Training hyperparameters
tf.app.flags.DEFINE_integer("max_steps", 15000, "Steps until training loop stops.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")

tf.app.flags.DEFINE_boolean("exponential_decay", False, "Whether to use exponential decay.")
tf.app.flags.DEFINE_float("decay_steps", 4000, "Number of steps for learning rate to decay by decay_rate")
tf.app.flags.DEFINE_boolean("staircase", True, "Whether staircase decay (use of integer division in decay).")
tf.app.flags.DEFINE_float("decay_rate", 0.75, "Learning rate.")

tf.app.flags.DEFINE_boolean("clip_gradients", True, "Whether to clip gradients.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")

# Model hyperparameters
tf.app.flags.DEFINE_string("model", 'dcnplus', "Model to train or evaluate, dcnplus / baseline")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("trainable_initial_state", False, "Make RNNCell initial states trainable.")  # Not implemented
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("trainable_embeddings", False, "Make embeddings trainable.")

# DCN+ hyperparameters
tf.app.flags.DEFINE_float("input_keep_prob", 0.975, "Encoder: Fraction of units randomly kept of inputs to RNN.")
tf.app.flags.DEFINE_float("output_keep_prob", 0.85, "Encoder: Fraction of units randomly kept of outputs from RNN.")
tf.app.flags.DEFINE_float("state_keep_prob", 0.85, "Encoder: Fraction of units randomly kept of encoder states in RNN.")
tf.app.flags.DEFINE_integer("pool_size", 4, "Number of units the maxout network pools.")
tf.app.flags.DEFINE_integer("max_iter", 4, "Maximum number of iterations of decoder.")
tf.app.flags.DEFINE_float("keep_prob", 0.85, "Decoder: Fraction of units randomly kept on non-recurrent connections.")

# Data hyperparameters
tf.app.flags.DEFINE_integer("max_question_length", 25, "Maximum question length.")
tf.app.flags.DEFINE_integer("max_paragraph_length", 400, "Maximum paragraph length and the output size of your model.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")  # Not used

# Evaluation arguments
tf.app.flags.DEFINE_integer("eval_size", 100, "Number of samples to use for evaluation.")

# Directories etc.
tf.app.flags.DEFINE_string("model_name", datetime.now().strftime('%y%m%d_%H%M%S'), "Models name, used for folder management.")
tf.app.flags.DEFINE_string("data_dir", os.path.join("..", "data", "squad"), "SQuAD directory (default ../data/squad)") 
tf.app.flags.DEFINE_string("train_dir", os.path.join("..", "checkpoints"), "Training directory to save the model parameters (default: ../checkpoints).")
tf.app.flags.DEFINE_integer("print_every", 50, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("vocab_path", os.path.join("..", "data", "squad", "vocab.dat"), "Path to vocab file (default: ../data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ../data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS

# TODO implement batch evaluation
# TODO hyperparameter random search
# TODO add shell
# TODO implement early stopping, or reload
# TODO implement EM
# TODO Write final Dev set eval to a file that's easily inspected

def exact_match(prediction, truth):
    pass

def do_eval(model, train, dev, eval_metric):
    checkpoint_dir = os.path.join(FLAGS.train_dir, FLAGS.model_name, 'eval')

    # Parameter space size information
    num_parameters = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
    logging.info('Number of parameters %d' % num_parameters)
    for v in tf.trainable_variables():
        logging.info(f'Variable {v} has {v.get_shape().num_elements()} entries')
    # TODO test if answer_span placeholder is still necessary without monitoredtrainingsesion

    saver = tf.train.Saver()
    # TODO add loop to run over all checkpoints in eval folder, 
    # Training session
    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))

        # Train/Dev Evaluation
        start_evaluate = timer()
        train_f1 = eval_metric(session, model, train, size=FLAGS.eval_size)
        dev_f1 = eval_metric(session, model, dev, size=FLAGS.eval_size)
        logging.info(f'Train/Dev F1: {train_f1:.3f}/{dev_f1:.3f}')
        logging.info(f'Time to evaluate: {timer() - start_evaluate:.1f} sec')

def do_train(model, train, dev, eval_metric):
    checkpoint_dir = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    
    hooks = [
        tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
        tf.train.NanTensorHook(model.loss)
    ]

    # Parameter space size information
    num_parameters = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
    logging.info('Number of parameters %d' % num_parameters)
    for v in tf.trainable_variables():
        logging.info(f'Variable {v} has {v.get_shape().num_elements()} entries')

    losses = []

    # Training session  
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           checkpoint_dir=checkpoint_dir, 
                                           save_summaries_steps=20) as session:
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

def main(_):
    # Load data
    train = SquadDataset(*get_data_paths(FLAGS.data_dir, name='train'), 
                         max_question_length=FLAGS.max_question_length, 
                         max_paragraph_length=FLAGS.max_paragraph_length)
    dev = SquadDataset(*get_data_paths(FLAGS.data_dir, name='val'), 
                         max_question_length=FLAGS.max_question_length, 
                         max_paragraph_length=FLAGS.max_paragraph_length) # probably not cut
    # TODO convert to TF Dataset API
    # train = tf.convert_to_tensor(train)
    # dev = tf.convert_to_tensor(dev)
    # tf.contrib.data.Dataset()

    logging.info(f'Train/Dev size {train.length}/{dev.length}')

    # Load embeddings
    embed_path = FLAGS.embed_path or pjoin(FLAGS.data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = np.load(embed_path)['glove']  # 115373
    # vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    # vocab, rev_vocab = initialize_vocab(vocab_path) # dict, list
    
    is_training = (FLAGS.mode == 'train' or FLAGS.mode == 'overfit')
    
    # Build model
    if FLAGS.model == 'dcnplus':
        model = DCNPlus(embeddings, FLAGS.__flags, is_training=is_training)
    elif FLAGS.model == 'baseline':
        model = Baseline(embeddings, FLAGS.__flags)
    elif FLAGS.model == 'cat':
        model = Graph(embeddings, is_training=is_training)
    else:
        raise ValueError(f'{FLAGS.model} is not a supported model')
    
    # Run mode
    if FLAGS.mode == 'train':
        save_flags()
        do_train(model, train, dev, evaluate)
    elif FLAGS.mode == 'eval':
        do_eval(model, train, dev, evaluate)
    elif FLAGS.mode == 'overfit':
        test_overfit(model, train, evaluate)
    else:
        raise ValueError(f'Incorrect mode entered, {FLAGS.mode}')


def save_flags():
    model_path = os.path.join(FLAGS.train_dir, FLAGS.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    json_path = os.path.join(FLAGS.train_dir, FLAGS.model_name, "flags.json")
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump(FLAGS.__flags, f, indent=4)


def test_overfit(model, train, evaluate):
    """
    Tests that model can overfit on small datasets.
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
                    f1 = evaluate(session, model, train, size=test_size)
                    print(f'F1: {f1:.2f}')
            global_step = tf.train.get_global_step().eval()
            print(f'Epoch took {timer() - epoch_start:.2f} s (step: {global_step})')

if __name__ == "__main__":
    #test_overfit()
    tf.app.run()
