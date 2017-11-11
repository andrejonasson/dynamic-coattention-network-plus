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
from model import QASystem
from baseline import encode, decode
from dataset import SquadDataset

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string("model_name", datetime.now().strftime('%y%m%d_%H%M%S'), "Models name, used for folder management.")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")  # 0.005
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 300, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")

tf.app.flags.DEFINE_string("data_dir", os.path.join("..", "data", "squad"), "SQuAD directory (default ../data/squad)") 
tf.app.flags.DEFINE_string("train_dir", os.path.join("..", "checkpoints"), "Training directory to save the model parameters (default: ../checkpoints).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", os.path.join("..", "log"), "Path to store log and flag files (default: ../log)")
tf.app.flags.DEFINE_integer("print_every", 50, "How many iterations to do per print.")
tf.app.flags.DEFINE_string("vocab_path", os.path.join("..", "data", "squad", "vocab.dat"), "Path to vocab file (default: ../data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ../data/squad/glove.trimmed.{embedding_size}.npz)")

# TODO add flag for what model should be used
# TODO tf.train.polynomial_decay
# TODO implement early stopping, or reload
# TODO implement EM
# TODO make framework compatible with VM Image on GCLOUD
# TODO write all hyperparams to checkpoints folder, write final Dev set eval to a file that's easily inspected


FLAGS = tf.app.flags.FLAGS

def exact_match(prediction, truth):
    pass

def do_train(model, train, dev, eval_metric):
    # if hasattr(model, 'graph'):
    #     graph = model.graph
    # else:
    #     graph = tf.get_default_graph()
    
    # how big a sample to evaluate on
    eval_size = 100
    steps = 5000

    checkpoint_dir = os.path.join(FLAGS.train_dir, FLAGS.model_name)

    # Two writers needed to enable plotting two lines in one plot
    dev_summary_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'dev'))
    train_summary_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'train'))
    
    hooks = [
        tf.train.StopAtStepHook(last_step=steps),
        tf.train.NanTensorHook(model.loss),
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
            if step == 10 or step == 100 or step % FLAGS.print_every == 0:
                mean_loss = sum(losses)/len(losses)
                losses = []
                print(f'Step {step}, loss {mean_loss:.2f}')
            
            # Train/Dev Evaluation
            if eval_metric is not None and step != 0 and (step == 50 or step == 200 or step % 500 == 0):
                start_evaluate = timer()
                train_f1 = eval_metric(session, model, train, size=eval_size)
                train_summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='f1', simple_value=train_f1)]),
                    step
                )
                dev_f1 = eval_metric(session, model, dev, size=eval_size)
                dev_summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='f1', simple_value=dev_f1)]),
                    step
                )
                logging.info(f'Step {step}, Train/Dev F1: {train_f1:.3f}/{dev_f1:.3f}')
                logging.info(f'Step {step}, Time to evaluate: {timer() - start_evaluate:.1f} sec')
            
            # Final evaluation on full development set
            if step == steps:
                # TODO need to change Dev to full ~(700 paragraph length, 100 question length)
                start_evaluate = timer()
                dev_f1 = evaluate(session, model, dev, size=dev.length)
                logging.info(f'Train/Dev F1: /{dev_f1:.3f}')  #{train_f1:.3f}
                logging.info(f'Time to evaluate full train/dev set: {timer() - start_evaluate:.1f} sec')

def main(_):
    # Load data
    data_hparams = {
        'max_paragraph_length': FLAGS.output_size,
        'max_question_length': 25
    }
    train = SquadDataset(*get_data_paths(FLAGS.data_dir, name='train'), **data_hparams)
    dev = SquadDataset(*get_data_paths(FLAGS.data_dir, name='val'), **data_hparams)  # probably not cut
    
    # TODO convert to TF Dataset API
    # train = tf.convert_to_tensor(train)
    # dev = tf.convert_to_tensor(dev)
    # tf.contrib.data.Dataset()

    logging.info(f'Train/Dev size {train.length}/{dev.length}')

    (questions, paragraphs, question_lengths, paragraph_lengths, answers) = train[:]
    # Load embeddings
    embed_path = FLAGS.embed_path or pjoin(FLAGS.data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = np.load(embed_path)['glove']  # 115373
    # vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    # vocab, rev_vocab = initialize_vocab(vocab_path) # dict, list
    
    # Build model

    test_hparams = {
        'learning_rate': FLAGS.learning_rate,
        'keep_prob': 1.0,
        'trainable_embeddings': False,
        'clip_gradients': True,
        'max_gradient_norm': FLAGS.max_gradient_norm,
        'state_size': FLAGS.state_size,

    }

    model = QASystem(encode, decode, embeddings, test_hparams)
    #model = Graph(embeddings, is_training=True)

    do_train(model, train, dev, evaluate)
    # if not os.path.exists(FLAGS.log_dir):
    #     os.makedirs(FLAGS.log_dir)
    # file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    # logging.getLogger().addHandler(file_handler)

    # print(vars(FLAGS))
    # with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
    #     json.dump(FLAGS.__flags, fout)
        #save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        #qa.train(sess, dataset, save_train_dir)

        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

def test_overfit():
    data_hparams = {
        'max_paragraph_length': 300,
        'max_question_length': 25
    }
    train = SquadDataset(*get_data_paths(FLAGS.data_dir, name='train'), **data_hparams)
    dev = SquadDataset(*get_data_paths(FLAGS.data_dir, name='val'), **data_hparams)  # probably not cut

    embed_path = FLAGS.embed_path or pjoin(FLAGS.data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = np.load(embed_path)['glove']  # 115373

    test_hparams = {
        'learning_rate': 0.01,
        'keep_prob': 1.0,
        'trainable_embeddings': False,
        'clip_gradients': True,
        'max_gradient_norm': 5.0
    }
    model = QASystem(encode, decode, embeddings, test_hparams)
    
    epochs = 100
    test_size = 32
    steps_per_epoch = 10
    train.question, train.paragraph, train.question_length, train.paragraph_length, train.answer = train[:test_size]
    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_start = timer()
            for step in range(steps_per_epoch):
                loss, _ = model.training_step(sess, *train[:test_size])
                if (step == 0 and epoch == 0):
                    print(f'Entropy - Result: {loss:.2f}, Expected (approx.): {2*np.log(FLAGS.output_size):.2f}')
                if step == steps_per_epoch-1:
                    print(f'Cross entropy: {loss}')
                    train.length = 32
                    print(evaluate(sess, model, train, size=test_size))
            global_step = tf.train.get_global_step().eval()
            print(f'Epoch took {timer() - epoch_start:.2f} s (step: {global_step})')

if __name__ == "__main__":
    #test_overfit()
    tf.app.run()
