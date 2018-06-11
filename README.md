# Active learning for deep reference parsing

Master's semester project (8 ECTS) at the École polytechnique fédérale de Lausanne (EPFL). <br />
Author: Mattia Martinelli. <br />
Supervisor: Giovanni Colavizza. <br />
Date: 08/06/2018. <br />
The project has been conducted at the [Digital Humanities Laboratory](https://dhlab.epfl.ch/) (DHLAB) @ EPFL. <br />


## Abstract
In recent years, models based on deep neural networks (DNNs) have shown outstanding results in several domains, outperforming other machine learning techniques. However, such models often require large collections of labeled data to achieve state-of-the-art performance. Labeling data can be a costly and time-consuming process, and some research has been conducted on reducing training
sets in a methodical manner. In this work, we show that an accuracy close to that of a fully-trained model can be achieved with only a fraction of the training data. We develop a lightweight CNN-CNN-LSTM model, consisting of convolutional (CNN) character and word encoders, and a long short term memory (LSTM) tag decoder; and we use it to parse bibliographic references from humanities literature. Active learning techniques are explored, and in particular we present an approach based on uncertainty sampling. We are able to nearly achieve the accuracy of our best models on two tasks with, respectively, 40% and 19% of the original training data.

**Keywords**: Reference parsing, deep learning, active learning, uncertainty sampling.


## Dataset

We work on a dataset of bibliography references from humanities literature, which can be downloaded [here](https://github.com/dhlab-epfl/LinkedBooksReferenceParsing). <br />
The dataset is provided as a JSON file. It needs to be decompressed into three separeted datasets: train, test, and valid.

Our models also uses pretrained vectors (trained with Word2vec), which can be downloaded here.

Further information about dataset and vectors can be found in the report.


## Repository structure

This repository is structured as follows:

- code:
  - models.py: contains the two Keras models (BiLSTM and CNN-CNN-LSTM).
  - active.py: active learning functions and main algorithm.
  - utils.py: support functions for the models.
  - general.py: script with functions to convert bibtex files into simil-CoNLL datasets.
- runActiveLearning.py: script to run our active learning model.
- runCNN.py: script to run our CNN model only.
- MS_Project_Martinelli.pdf: project report.


## Setup

Our code has been written in Python 3 (v3.5.4). You also need the following modules:

- Anaconda 4.3
- Tensorflow 1.4.0
- Keras 2.1.6
- Keras-contrib 0.0.2


## Credits and license
The code is distributed under MIT license. Third-party components are subject to their respective licenses.  <br />
Special thanks to [Danny Rodrigues Alves](https://github.com/RA-Danny) for the BiLSTM model and utils scripts. <br />


