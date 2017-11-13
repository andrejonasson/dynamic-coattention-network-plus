Question Answering / Reading Comprehension
==========================================

Work in progress.

The project will include some reimplementations of papers on coattention mechanisms. 

Networks implemented: A simple baseline model (BiLSTM + Coattention + Naive decoder) and a partial Dynamic Coattention Network Plus (DCN+) (work in progress).

## Introduction

SQuAD (Stanford Question Answering Dataset)[3][4] formulates a machine learning problem where the model receives a question and a passage and is tasked with answering the question using the passage. The answers are limited to spans of text. The training data consists of (question, paragraph, answer span) triplets. Due to the nature of the task, combining the information contained in the passage with the question posed is paramount to achieve good performance. (See references for more information)

Recurrent neural networks that combine the information from the question and paragraph using coattention mechanisms such as [1] and [2] have achieved the best results in the competition so far. This project aims to reimplement some of these architectures in TensorFlow and achieve competitive results on the SQuAD dataset.

## Networks

### Baseline model
Baseline model achieves ~0.46 F1 (limited to paragraphs below 300 words and questions below 25 words) on the development set after testing a few hyperparameters.

Best hyperparameters
```
Steps = 15000
Word embedding size = 100
Hidden state size = 100
Optimizer = Adam
Learning Rate = 0.01
Decay = Exponential (Staircase)
Decay Steps = 4500
Decay Rate = 0.5

Dev F1 = ~0.46 (300 max length paragraph, 25 max length questions)
```
Increasing embedding size and state size should improve performance further.

### Dynamic Coattention Network Plus (DCN+)
The project contains a complete implementation of the DCN+ encoder. Decoder is in progress. 

Encoder without sentinel with naive decoder achieves ~0.60 Dev F1 with similar states and sizes as Baseline model above.

Instead of mixed objective the implementation will have cross entropy.

## Dependencies

The project has only been tested for Python 3.6 with TensorFlow 1.3. Support for prior versions will not be added.

## Instructions

Move under the project folder (the one containing the README.md)

1. Install the requirements (you may want to create and activate a virtualenv)
``` sh
pip -r requirements.txt
```

2. To download squad run
``` sh
$ wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P download/squad/
```
Download punkt if needed
``` sh
python -m nltk.downloader punkt
```
then preprocess SQuAD using
```
python preprocessing/squad_preprocess.py
```
While the preprocessing is running you can continue with Step 3 in another terminal in the same folder. 

3. Issue the command
``` sh
$ wget http://nlp.stanford.edu/data/glove.6B.zip -P download/dwr/
```
to download Wikipedia 100/200/300 dimensional GLoVe word embeddings or
``` sh
$ wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P download/dwr/
```
for Common Crawl 300 dimensional GLoVe word embeddings

Extract the Wikipedia embeddings
``` sh
$ tar -xvzf download/dwr/glove.6B.zip --directory download/dwr/
```
or the common crawl embeddings
``` sh
$ tar -xvzf download/dwr/glove.42B.300d.zip --directory download/dwr/
```
4. Change directory to the one containing the code (`qa_data.py` etc.) and then when Step 2 and 3 are done run
``` sh
$ python qa_data.py --glove_dim EMBEDDINGS_DIMENSIONS --glove_source SOURCE
```
replacing `EMBEDDINGS_DIMENSIONS` by the word embedding size you want (100, 200, 300) and `SOURCE` by 'wiki' if using Wikipedia embeddings and 'crawl' if using common crawl embeddings (you may omit `--glove_dim` if you choose 'crawl'). `qa_data.py` will process the embeddings and create a 95-5 split of the training data where the 95% will be used as a training set and the rest is a development set.

Once complete run (Additionally, you may need to comment out the line importing `cat.py` in `train.py`.)
```
python train.py
```
to train the network. During training checkpoints and logs will be placed under a timestamped folder in checkpoints folder.

For Tensorboard, run
```
tensorboard --logdir ../checkpoints
```
The F1 on train/dev using a sample (~400 by default), gradient norm, learning rate and computational graph should be present among other metrics.

## Acknowledgements

The project uses code from Stanford's CS224n to read and transform the original SQuAD dataset together with the GLoVe vectors to an appropriate format for model development. These files or functions have been annotated with "CS224n" at the beginning.

## References

[1] Dynamic Coattention Networks For Question Answering, Xiong et al, https://arxiv.org/abs/1611.01604

[2] DCN+: Mixed Objective and Deep Residual Coattention for Question Answering, Xiong et al, https://arxiv.org/abs/1711.00106

[3] SQuAD: 100,000+ Questions for Machine Comprehension of Text, Rajpurkar et al, https://arxiv.org/abs/1606.05250

[4] https://rajpurkar.github.io/SQuAD-explorer/
