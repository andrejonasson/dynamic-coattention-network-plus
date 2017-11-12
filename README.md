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

F1 = ~0.46 (300 max length paragraph, 25 max length questions)
```

### Dynamic Coattention Network Plus (DCN+)
The project a nearly complete implementation of the DCN+ encoder (work in progress). 

Instead of mixed ojective the implementation will have cross entropy.

## Dependencies

The project has only been tested for Python 3.6 with TensorFlow 1.3. Support for prior versions will not be added.

## Running

``` sh
$ python train.py
```

## Acknowledgements

The project uses code from Stanford's CS224n to read and transform the original SQuAD dataset together with the GLoVe vectors to an appropriate format for model development. These files or functions have been annotated with 'CS224n' at the beginning.

## References

[1] Dynamic Coattention Networks For Question Answering, Xiong et al, https://arxiv.org/abs/1611.01604

[2] DCN+: Mixed Objective and Deep Residual Coattention for Question Answering, Xiong et al, https://arxiv.org/abs/1711.00106

[3] SQuAD: 100,000+ Questions for Machine Comprehension of Text, Rajpurkar et al, https://arxiv.org/abs/1606.05250

[4] https://rajpurkar.github.io/SQuAD-explorer/
