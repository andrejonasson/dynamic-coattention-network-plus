Dynamic Coattention Network Plus - Question Answering
=====================================================

## Introduction

SQuAD (Stanford Question Answering Dataset)[3][4] formulates a machine learning problem where the model receives a question and a passage and is tasked with answering the question using the passage. The answers are limited to spans of text. The training data consists of (question, paragraph, answer span) triplets. Due to the nature of the task, combining the information contained in the passage with the question posed is paramount to achieve good performance (See references for more information). Recurrent neural networks that combine the information from the question and paragraph using coattention mechanisms such as the Dynamic Coattention Network [1] and its deeper and improved version [2] have achieved the best results in the task so far.

## Networks

### Baseline model
Baseline model (BiLSTM + DCN-like Coattention + Naive decoder).

Starting point for hyperparameters
```
Steps = ~20000/40000
Word embedding size = 100
Hidden state size = 100
Optimizer = Adam
Batch size = 64
RNN Input Dropout = 15%
Learning Rate = 0.005/0.001

Achieves dev ~ F1 0.620 / EM 0.452
```

### Dynamic Coattention Network (DCN)
Implemented using components from DCN+ but with one layer attention and a single directional initial LSTM encoder.

### Dynamic Coattention Network Plus (DCN+)

DCN+ encoder combines the question and passage using a dot-product based coattention mechanism, similar to the attention in the Transformer Network, Vaswani et al (2017). The decoder is application specific, specifically made for finding an answer span within a passage, it uses an iterative mechanism for recovering from local minima. Instead of a mixed objective the implementation uses cross entropy like the vanilla DCN.

<img src="encoder.png">

For the implementation see `networks.dcn_plus`. An effort has been made to document each component. Each component of the encoder (coattention layer, affinity softmax masking, sentinel vectors and encoder units) and certain parts of the decoder are modular and can easily be used with other networks.

### Todos
- Character embeddings (Char-CNN as in BiDAF [5])
- Sparse mixture of experts
- Compatibility with official evaluation script
- Figure out where and to what degree dropout should be applied

## Additional Modules

### Linear time maximum probability product answer span
The project contains an implementation of the dynamic programming approach for finding the maximum product of the start and end probabilities as used by BiDAF and others. The implementation is inspired by [bi-att-flow](https://github.com/allenai/bi-att-flow/blob/master/basic_cnn/evaluator.py), except that it acts on batches instead of single examples and is in TensorFlow.

## Instructions

### Dependencies

The project requires Python 3.6 with TensorFlow 1.4. Support for prior versions will not be added.

### Getting started

Move under the project folder (the one containing the README.md)

1. Install the requirements (you may want to create and activate a virtualenv)
``` sh
$ pip install -r requirements.txt
```

2. Download punkt if needed
``` sh
$ python -m nltk.downloader punkt
```
then download and preprocess SQuAD using
``` sh
$ python question_answering/preprocessing/squad_preprocess.py
```
While the preprocessing is running you can continue with Step 3 in another terminal. 

3. Issue the command
``` sh
$ python question_answering/preprocessing/dwr.py <GLOVE_SOURCE>
```
to download and extract GLoVe embeddings, where `<GLOVE_SOURCE>` is either `wiki` for Wikipedia 100/200/300 dimensional GLoVe word embeddings (~800mb) or `crawl_ci`/`crawl_cs` for Common Crawl 300 dimensional GLoVe word embeddings (~1.8-2.2gb) where `crawl_ci` is the case insensitive version. Note that at a later step Common Crawl requires at least 4 hours of processing while Wikipedia 100 dimensional GLoVE finishes in about half an hour.

4. When Step 2 and 3 are complete change directory to the folder containing the code (`main.py` etc.) and run
``` sh
$ python preprocessing/qa_data.py --glove_dim <EMBEDDINGS_DIMENSIONS> --glove_source <GLOVE_SOURCE>
```
replacing `<EMBEDDINGS_DIMENSIONS>` by the word embedding size you want (100, 200 or 300) and `<GLOVE_SOURCE>` by the embedding chosen above. `qa_data.py` will process the embeddings and create a 95-5 split of the training data where the 95% will be used as a training set and the rest as a development set.

### Usage

The default mode of `main.py` is to train a DCN+ network, run
``` sh
$ python main.py --embedding_size <EMBEDDINGS_DIMENSIONS>
```
to begin training. See the source code for all the arguments and modes that `main.py` supports. Checkpoints and logs will be placed under a timestamped folder in the `../checkpoints` folder by default. 

### Interactive Shell

To see a trained model in action, load your model in mode `shell` and ask it questions about passages. 

<img src="shell.png">

### Tensorboard
For Tensorboard, run
``` sh
$ tensorboard --logdir checkpoints
```
from the project folder and navigate to `localhost:6006`. The gradient norm and learning rate should be present among other metrics. The computational graph can also be viewed.

## Acknowledgements

The project uses code from Stanford's CS224n to read and transform the original SQuAD dataset together with the GLoVe vectors to an appropriate format for model development. These files or functions have been annotated with "CS224n" at the beginning.

## References

[1] Dynamic Coattention Networks For Question Answering, Xiong et al, https://arxiv.org/abs/1611.01604, 2016

[2] DCN+: Mixed Objective and Deep Residual Coattention for Question Answering, Xiong et al, https://arxiv.org/abs/1711.00106, 2017

[3] SQuAD: 100,000+ Questions for Machine Comprehension of Text, Rajpurkar et al, https://arxiv.org/abs/1606.05250, 2016

[4] https://rajpurkar.github.io/SQuAD-explorer/

[5] Bidirectional Attention Flow for Machine Comprehension, Seo et al, https://arxiv.org/abs/1611.01603, 2016

## Author
André Jonasson / [@andrejonasson](https://github.com/andrejonasson)