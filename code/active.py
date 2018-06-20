# -*- coding: utf-8 -*-

"""
    DHLAB - IC - EPFL
    
    This file contains functions for uncertainty sampling active learning.
    Important note for fist-time users: this file is an extension of models.py and utils.py. Please check them out first!
    
    author: Mattia Martinelli
    date: 08/06/2018
    
"""

# Modules
import os
import random
import numpy as np
import tensorflow

# Keras function
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Input, TimeDistributed, Flatten, Convolution1D, MaxPooling1D, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras_contrib.utils import save_load_utils

from models import *
from utils import *

from scipy.stats import rankdata
import scipy as sc
from shutil import copyfile
import operator
from collections import defaultdict
import time


def BiLSTM_score(filename, X_w, X_i, y, word2ind, maxWords, ind2label,
              word_embeddings=True, pretrained_embedding="", word_embedding_size=100,
              maxChar=0, char_embedding_type="", char2ind="", char_embedding_size=50,
              lstm_hidden=32, dropout=0.5, optimizer='rmsprop',
              train = False, X_train = None, y_train = None, X_test = None, y_test = None,
              nbr_epochs = 1, batch_size=128, early_stopping_patience=-1,
              folder_path="BiLSTM_results", score_name = "uncertainty_scores", print_to_file = True
            ):   
    """
        The function computes, for each input token, three uncertainty sampling scores: probability, margin, entropy.
        Scores are generated with the BiLSTM model. Softmax is the prediction function.
        Detailed information about how scores are computed can be found in the report.
        The underlying model can be trained. Otherwise it uses the weights in "folder_path/filename/filename.h5".
        
        The result is stored in a csv file, structured with the following columns:
        "Sequence", "Position", "Token", "Target", "Predicted", "Posterior", "Confidence", "Margin", "Entropy", "Reference"
        where:
            - Sequence is the reference index in the dataset (0 first reference, 1 second reference, ...)
            - Position is the index of the token in the reference, i.e. the index in the reference (0 first token, 1 second token, ...)
            - Token is the token.
            - Target is the real label of the token.
            - Predicted is the actual prediction on the token.
            - Posterior is the output probability of prediction.
            - Confidence is the confidence (i.e. posterior) ranking over all tokens.
            - Margin is the margin raking over all tokens.
            - Entropy is the entropy ranking over all tokens.
            - Reference is the full reference to which the token belongs as a string.
        Example:
            "Sequence", "Position", "Token", "Target", "Predicted", "Posterior", "Confidence", "Margin", "Entropy", "Reference"
                    4           0    Maria    Author       Author         99.4            10        9          11    "Maria and ..."
                    
        
        :param filename: File to redirect the printing.
        :param X_w: Data to score, given in the original word format of load_data function (in utils.py).
        :param X_i: Data to score, given in the indexed format of encodePadData_x function (in utils.py).
        :param y: Labels of the data to score, given in the original word format of load_data function (in utils.py).
        :param folder_path: Path to the directory storing all to-be-generated files and folders.
        :param print_to_file: if True redirects the printings to a file (given in filename), if False std_out is kept
        :param score_name: Name of the file with the scores.
        :see Please for the other parameters refer to BiLSTM_model function in models.py (identical parameters are named the same).
        
        :return void
               
    """
    
    # Where model weights will be stored
    filepath = folder_path+"/"+filename+"/"+filename
    best_model_weights_path = "{0}.h5".format(filepath)

    # Get compiled  model with input parameters, if needed it can be trained too
    model = BiLSTM_model( filename = filename, train = train, output =  "softmax",
              X_train = X_train, X_test= X_test, word2ind = word2ind, maxWords = maxWords,
              y_train = y_train, y_test = y_test, ind2label = ind2label,
              validation = False, X_valid = None, y_valid = None,
              word_embeddings = word_embeddings, pretrained_embedding = pretrained_embedding, word_embedding_size = word_embedding_size,
              maxChar = maxChar, char_embedding_type = char_embedding_type, char2ind = char2ind, char_embedding_size = char_embedding_size,
              lstm_hidden = lstm_hidden, batch_size = batch_size, dropout = dropout, optimizer = optimizer, 
              nbr_epochs = nbr_epochs, early_stopping_patience = early_stopping_patience,
              folder_path = folder_path, gen_confusion_matrix = False, return_model = True, print_to_file = print_to_file
            )   
            
    # HACK: optmizer weight length issue
    # https://github.com/keras-team/keras/issues/4044
    import h5py
    with h5py.File(best_model_weights_path, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    save_load_utils.load_all_weights(model, best_model_weights_path)

    # Compute predictions and uncertainty scores
    probs = model.predict(X_i)
    probs = np.asarray(probs)

    # Arguments of sorted values in ascending order
    pred_sort_index = np.argsort(probs)
    # Reverse the one-hot encoding
    #true_index = np.argsort(y_target)[:,:,-1]
    true_index = np.argmax(y, axis=-1) 

    # Predict probability of best prediction
    pred_index = pred_sort_index[:,:,-1]
    grid = np.indices((pred_index.shape[0], pred_index.shape[1]))   
    pred_prob = probs[grid[0],grid[1],pred_index[grid[0],grid[1]]]

    # Predict probability of second best prediction
    pred_2_index = pred_sort_index[:,:,-2]
    grid = np.indices((pred_2_index.shape[0], pred_2_index.shape[1]))   
    pred_2_prob = probs[grid[0],grid[1],pred_2_index[grid[0],grid[1]]]  

    # Margin score, computed as the difference between the best prediction 
    # and the second best prediction
    pred_margin = pred_prob - pred_2_prob 

    # Entropy score, compute as entropy over all prediction probabilities ù
    # The entropy score is inversed to compute rank later
    pred_entropy = np.sum(np.multiply(probs,np.log(probs)), axis = -1)

    # Index 0 in the predictions referes to padding
    ind2labelNew = ind2label[0].copy()
    ind2labelNew.update({0: "null"})

    # Compute the labels for each prediction
    pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
    true_label = [[ind2labelNew[x] for x in b] for b in true_index]

    # Flatten to uniform with ranking
    pred_flat = np.ravel(pred_label)
    true_flat = np.ravel(true_label)
    prob_flat = np.ravel(pred_prob)

    # Compute ranking
    prob_rank = rankdata(pred_prob, method='min')
    margin_rank = rankdata(pred_margin, method='min')
    entropy_rank = rankdata(pred_entropy, method='min')

    # Fill CSV rows
    rows = []
    seq_len = maxWords
    for i, seq in enumerate(X_w):
        # skip first sequence
        if i == 0: 
            continue
        seq_offset = i*maxWords + seq_len - len(seq)
        for j,w in enumerate(seq):
            index = seq_offset + j
            rows.append(
               (i, j + 1, w, true_flat[index], pred_flat[index], 
                round(prob_flat[index], 4), prob_rank[index], 
                margin_rank[index], entropy_rank[index], 
                " ".join(str(s) for s in seq))
            )

    # Write results on CSV file
    columns = ("Sequence", "Position", "Token", "Target", "Predicted", "Posterior", 
                "Confidence", "Margin", "Entropy", "Reference")
    
    # Store scores
    filename="score"
    os.makedirs(folder_path+"/"+filename, exist_ok=True)
    score_result_path = folder_path+"/"+filename+"/"+score_name
    write_to_csv(score_result_path, columns, rows)   
    
    print("BiLSTM score has terminated!")
    
    
def CNN_score(filename, X_w, X_i, y, word2ind, maxWords, ind2label, maxChar, char2ind, 
              pretrained_embedding="", word_embedding_size=100,
              char_embedding_size=50, lstm_hidden=32, dropout=0.5, optimizer='rmsprop',
              folder_path="CNN_results", score_name = "scores", print_to_file = True,
              train = False, X_train = None, y_train = None, X_test = None, y_test = None,
              nbr_epochs = 5, batch_size=128, early_stopping_patience=-1):   
    """
        The function computes, for each input token, three uncertainty sampling scores: probability, margin, entropy.
        Scores are generated with the CNN-CNN-LSTM model.
        Detailed information about how scores are computed can be found in the report.
        The underlying model can be trained. Otherwise it uses the weights in "folder_path/filename/filename.h5".
        The result is stored in a csv file. For additional information on the output file, please refer to the BiLSTM_score function description.
        
        :param filename: File to redirect the printing.
        :param X_w: Data to score, given in the original word format of load_data function (in utils.py).
        :param X_i: Data to score, given in the indexed format of encodePadData_x function (in utils.py).
        :param y: Labels of the data to score, given in the original word format of load_data function (in utils.py).
        :param folder_path: Path to the directory storing all to-be-generated files and folders.
        :param score_name: Name of the file with the scores.
        :param print_to_file: if True redirects the printings to a file (given in filename), if False std_out is kept
        :see Please for the other parameters refer to BiLSTM_model function in models.py (identical parameters are named the same).
        
        :return void
             
    """ 
    
    # Where model weights will be stored
    filepath = folder_path+"/"+filename+"/"+filename
    best_model_weights_path = "{0}.h5".format(filepath)

    # Get compiled model with input parameters, if needed it can be trained too
    model = CNN_model(filename = filename, train = train, 
              X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, 
              word2ind = word2ind, maxWords = maxWords, ind2label = ind2label, maxChar = maxChar, char2ind = char2ind, 
              validation=False, X_valid=None, y_valid=None,
              pretrained_embedding = pretrained_embedding, word_embedding_size = word_embedding_size, char_embedding_size = char_embedding_size,
              lstm_hidden = lstm_hidden, nbr_epochs = nbr_epochs, batch_size = batch_size, dropout = dropout, 
              optimizer= optimizer, early_stopping_patience=-1,
              folder_path=folder_path, gen_confusion_matrix=False, return_model = True, print_to_file = print_to_file
             )
        
    # HACK: optmizer weight length issue
    # https://github.com/keras-team/keras/issues/4044
    import h5py
    with h5py.File(best_model_weights_path, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    save_load_utils.load_all_weights(model, best_model_weights_path)

    # Compute predictions and uncertainty scores
    probs = model.predict(X_i)
    probs = np.asarray(probs)

    # Arguments of sorted values in ascending order
    pred_sort_index = np.argsort(probs)
    # Reverse the one-hot encoding
    #true_index = np.argsort(y_target)[:,:,-1]
    true_index = np.argmax(y, axis=-1) 

    # Predict probability of best prediction
    pred_index = pred_sort_index[:,:,-1]
    grid = np.indices((pred_index.shape[0], pred_index.shape[1]))   
    pred_prob = probs[grid[0],grid[1],pred_index[grid[0],grid[1]]]

    # Predict probability of second best prediction
    pred_2_index = pred_sort_index[:,:,-2]
    grid = np.indices((pred_2_index.shape[0], pred_2_index.shape[1]))   
    pred_2_prob = probs[grid[0],grid[1],pred_2_index[grid[0],grid[1]]]  

    # Margin score, computed as the difference between the best prediction 
    # and the second best prediction
    pred_margin = pred_prob - pred_2_prob 

    # Entropy score, compute as entropy over all prediction probabilities ù
    # The entropy score is inversed to compute rank later
    pred_entropy = np.sum(np.multiply(probs,np.log(probs)), axis = -1)

    # Index 0 in the predictions referes to padding
    ind2labelNew = ind2label[0].copy()
    ind2labelNew.update({0: "null"})

    # Compute the labels for each prediction
    pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
    true_label = [[ind2labelNew[x] for x in b] for b in true_index]

    # Flatten to uniform with ranking
    pred_flat = np.ravel(pred_label)
    true_flat = np.ravel(true_label)
    prob_flat = np.ravel(pred_prob)

    # Compute ranking
    prob_rank = rankdata(pred_prob, method='min')
    margin_rank = rankdata(pred_margin, method='min')
    entropy_rank = rankdata(pred_entropy, method='min')

    # Create CSV rows
    rows = []
    seq_len = maxWords
    for i, seq in enumerate(X_w):
        # skip first sequence
        if i == 0: 
            continue
        seq_offset = i*maxWords + seq_len - len(seq)
        for j,w in enumerate(seq):
            index = seq_offset + j
            rows.append(
               (i, j + 1, w, true_flat[index], pred_flat[index], 
                round(prob_flat[index], 4), prob_rank[index], 
                margin_rank[index], entropy_rank[index], 
                " ".join(str(s) for s in seq))
            )

    # Write results on CSV file
    columns = ("Sequence", "Position", "Token", "Target", "Predicted", "Posterior", 
                "Confidence", "Margin", "Entropy", "Reference")
    
    # Store scores
    filename="score"
    os.makedirs(folder_path+"/"+filename, exist_ok=True)
    score_result_path = folder_path+"/"+filename+"/"+score_name
    write_to_csv(score_result_path, columns, rows)   
    
    print("CNN score has terminated!")
    
    
    
def BiLSTM_query(filename, X_w, X_i, y, numSeqToQuery, mode, word2ind, maxWords, ind2label, query_seed = 42,
              write_to_disk = False, verbose= False, task = 1,
              word_embeddings=True, pretrained_embedding="", word_embedding_size=100,
              maxChar=0, char_embedding_type="", char2ind="", char_embedding_size=50,
              lstm_hidden=32, dropout=0.5, optimizer='rmsprop',
              train = False, X_train = None, y_train = None, X_test = None, y_test = None,
              nbr_epochs = 1, batch_size=128, early_stopping_patience=-1,
              folder_path="BiLSTM_results", print_to_file = True
            ):   
    """
        The function selects references from the dataset according to their entropy uncertainy sampling score.
        Scores are generated with the BiLSTM model. Softmax is the prediction function.
        Detailed information about how sequence scores are computed can be found in the report.
        The underlying can be trained. Otherwise it uses the weights in "folder_path/filename/filename.h5".
        
        :param filename: File to redirect the printing.
        :param X_w: Data to score, given in the original word format of load_data function (in utils.py).
        :param X_i: Data to score, given in the indexed format of encodePadData_x function (in utils.py).
        :param y: Labels of the data to score, given in the original word format of load_data function (in utils.py).
        :param numSeqToQuery: Number of references to query.
        :param mode: How references are queried:
                        - least: query least confident references, i.e. highest entropy.
                        - most: query most confident references, i.e. lowest entropy.
                        - random: query references randomly.
        :param query_seed: seed of the random sampling.
        :param write_to_disk: if True, stores in a text file the queried references.
        :param verbose: if True, and write_to_disk is True, store entropy scores and target label along with the tokens.
        :param task: for which task the function is querying. Effective only if write_to_disk is True. Must be a value between 1 and 3.
        :param folder_path: Path to the directory storing all to-be-generated files and folders.
        :param print_to_file: if True redirects the printings to a file (given in filename), if False std_out is kept
        :see Please for the other parameters refer to BiLSTM_model function in models.py (identical parameters are named the same).
        
        :return Indices of the queried references.
    """
    
    assert(task >= 1 and task <= 3)
    
    # Where model weights will be stored
    filepath = folder_path+"/"+filename+"/"+filename
    best_model_weights_path = "{0}.h5".format(filepath)

    # Get compiled model with input parameters, if needed it can be trained too
    model = BiLSTM_model( filename = filename, train = train, output =  "softmax",
              X_train = X_train, X_test= X_test, word2ind = word2ind, maxWords = maxWords,
              y_train = y_train, y_test = y_test, ind2label = ind2label,
              validation = False, X_valid = None, y_valid = None,
              word_embeddings = word_embeddings, pretrained_embedding = pretrained_embedding, word_embedding_size = word_embedding_size,
              maxChar = maxChar, char_embedding_type = char_embedding_type, char2ind = char2ind, char_embedding_size = char_embedding_size,
              lstm_hidden = lstm_hidden, batch_size = batch_size, dropout = dropout, optimizer = optimizer, 
              nbr_epochs = nbr_epochs, early_stopping_patience = early_stopping_patience,
              folder_path = folder_path, gen_confusion_matrix = False, return_model = True, print_to_file = print_to_file
            )   
        
    # HACK: optmizer weight length issue
    # https://github.com/keras-team/keras/issues/4044
    import h5py
    with h5py.File(best_model_weights_path, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    save_load_utils.load_all_weights(model, best_model_weights_path)

    # Compute predictions and uncertainty scores
    probs = model.predict(X_i)
    probs = np.asarray(probs)

    # Arguments of sorted values in ascending order
    pred_sort_index = np.argsort(probs)
    # Reverse the one-hot encoding
    #true_index = np.argsort(y_target)[:,:,-1]
    true_index = np.argmax(y, axis=-1) 
    
    # Predict probability of best prediction
    pred_index = pred_sort_index[:,:,-1]
    grid = np.indices((pred_index.shape[0], pred_index.shape[1]))   
    pred_prob = probs[grid[0],grid[1],pred_index[grid[0],grid[1]]]

    # Entropy score, compute as entropy over all prediction probabilities 
    # The entropy score is inversed to later sort the list in ascending order
    pred_entropy = np.sum(np.multiply(probs,np.log(probs)), axis = -1)
    
    # Entropy over the sequence (with no padding)
    # The value is computed as the average entropy w.r.t. the tokens which are not padding
    sequence_len = np.count_nonzero((X_i[0] != 0), -1)
    sequence_entropy = np.divide((pred_entropy * (X_i[0] != 0)).sum(axis = -1)[1:], sequence_len[1:])
    
    # Get indices of sorted array
    sequence_entropy_sort_index = np.argsort(sequence_entropy)
    # Add 1 to indices to take first line into account
    add1 = np.vectorize(lambda x: x + 1)
    sequence_entropy_sort_index = add1(sequence_entropy_sort_index)

    # Entropy over the sequence (with padding)
    # The value is computed as the sum of entropies of all tokens in the sequence
    #sequence_entropy = np.sum(pred_entropy, axis = -1)    
    #sequence_entropy_sort_index = np.argsort(sequence_entropy)
    
    # Index 0 in the predictions referes to padding
    ind2labelNew = ind2label[0].copy()
    ind2labelNew.update({0: "null"})

    # Compute the labels for each prediction
    pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
    true_label = [[ind2labelNew[x] for x in b] for b in true_index]

    # Flatten to uniform indexing
    pred_flat = np.ravel(pred_label)
    true_flat = np.ravel(true_label)
    entropy_flat = np.ravel(pred_entropy)
    
    seq_len = maxWords
    
    value_to_return = None
    
    # Get least confident (highest entropy)
    if mode == "least":
        query_index_least_rank = sequence_entropy_sort_index[:numSeqToQuery]
        value_to_return = query_index_least_rank
        filename="least.txt"
       
    # Get most confident (lowest entropy)
    if mode == "most":
        query_index_most_rank = sequence_entropy_sort_index[max((len(sequence_entropy_sort_index) - numSeqToQuery),0):]
        value_to_return = query_index_most_rank
        filename="most.txt"
        
    # Get random sequences
    if mode == "random":
        # Compute random indexing
        query_index_random = np.arange(1, len(sequence_entropy_sort_index) + 1)
        np.random.seed(query_seed)
        np.random.shuffle(query_index_random)
        query_index_random = query_index_random[:numSeqToQuery]
        value_to_return = query_index_random
        filename="random.txt"
        
    # NOTE: other uncertainty sampling measures can be inserted below.
          
    # Store sequences in a file
    if write_to_disk:
        os.makedirs(folder_path+"/"+"query_result", exist_ok=True)
        query_result_path = folder_path+"/"+"query_result"+"/"+filename
        with open(query_result_path, "w", encoding = "utf-8") as f:
            if verbose:
                f.write("token target predicted entropy\r\r")
            else:
                f.write("-DOCSTART- -X- -X- o\r\r")
            # Store least index rank
            for i in value_to_return: 
                seq = X_w[i]
                seq_offset = i*maxWords + seq_len - len(seq)
                for j,w in enumerate(seq):
                    index = seq_offset + j
                    if verbose:
                        f.write(w + " " + true_flat[index] + " " + pred_flat[index] + " " + str(entropy_flat[index]) + "\r")
                    else:
                        if task == 1:
                            f.write(w + " " + true_flat[index] + " o o\r")
                        elif task == 2:
                            f.write(w + " o " + true_flat[index] + " o\r")
                        elif task == 3:
                            f.write(w + " o o " + true_flat[index] + "\r")
                        else:
                            raise Exception('Bad task given.')
                f.write("\r")
          
    print("BiLSTM query has terminated!")
    return value_to_return
    
    
def CNN_query(filename, X_w, X_i, y, numSeqToQuery, mode, word2ind, maxWords, ind2label, maxChar, char2ind, seed = 42, 
              write_to_disk = False, verbose = False, task = 1,
              pretrained_embedding="", word_embedding_size=100, char_embedding_size=50, 
              lstm_hidden=32, dropout=0.5, optimizer='rmsprop',
              train = False, X_train = None, y_train = None, X_test = None, y_test = None,
              nbr_epochs = 5, batch_size=128, early_stopping_patience=-1, folder_path="CNN_results", print_to_file = True
             ):
    """
        The function selects references from the dataset according to their entropy uncertainy sampling score.
        Scores are generated with the CNN-CNN-LSTM model.
        Detailed information on how sequence scores are computed can be found in the report.
        The underlying model can be trained. Otherwise it uses the weights in "folder_path/filename/filename.h5".
        
        :param filename: File to redirect the printing.
        :param X_w: Data to score, given in the original word format of load_data function (in utils.py).
        :param X_i: Data to score, given in the indexed format of encodePadData_x function (in utils.py).
        :param y: Labels of the data to score, given in the original word format of load_data function (in utils.py).
        :param numSeqToQuery: Number of references to query.
        :param mode: How references are queried:
                        - least: query least confident references, i.e. highest entropy.
                        - most: query most confident references, i.e. lowest entropy.
                        - random: query references randomly.
                        - hybrid: hybrid least/most approach.
                        - Other methods can be added if needed.
        :param query_seed: seed of the random sampling.
        :param write_to_disk: if True, stores in a text file the queried references.
        :param verbose: if True, and write_to_disk is True, store entropy scores and target label along with the tokens.
        :param task: for which task the function is querying. Effective only if write_to_disk is True. Must be a value between 1 and 3.
        :param folder_path: Path to the directory storing all to-be-generated files and folders.
        :param print_to_file: if True redirects the printings to a file (given in filename), if False std_out is kept
        :see Please for the other parameters refer to BiLSTM_model function in models.py (identical parameters are named the same).
        
        :return indices of the queried references.
    """
    
    # Where model weights will be stored
    filepath = folder_path+"/"+filename+"/"+filename
    best_model_weights_path = "{0}.h5".format(filepath)

    # Get compiled model with input parameters, if needed it can be trained too
    model = CNN_model(filename = filename, train = train, 
              X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test, 
              word2ind = word2ind, maxWords = maxWords, ind2label = ind2label, maxChar = maxChar, char2ind = char2ind, 
              validation=False, X_valid=None, y_valid=None,
              pretrained_embedding = pretrained_embedding, word_embedding_size = word_embedding_size, char_embedding_size = char_embedding_size,
              lstm_hidden = lstm_hidden, nbr_epochs = nbr_epochs, batch_size = batch_size, dropout = dropout, 
              optimizer= optimizer, early_stopping_patience=-1,
              folder_path=folder_path, gen_confusion_matrix=False, return_model = True, print_to_file = print_to_file
             )
    
    # HACK: optmizer weight length issue
    # https://github.com/keras-team/keras/issues/4044
    import h5py
    with h5py.File(best_model_weights_path, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    save_load_utils.load_all_weights(model, best_model_weights_path)

    # Compute predictions and uncertainty scores
    probs = model.predict(X_i)
    probs = np.asarray(probs)

    # Arguments of sorted values in ascending order
    pred_sort_index = np.argsort(probs)
    # Reverse the one-hot encoding
    #true_index = np.argsort(y_target)[:,:,-1]
    true_index = np.argmax(y, axis=-1) 
    
    # Predict probability of best prediction
    pred_index = pred_sort_index[:,:,-1]
    grid = np.indices((pred_index.shape[0], pred_index.shape[1]))   
    pred_prob = probs[grid[0],grid[1],pred_index[grid[0],grid[1]]]

    # Entropy score, compute as entropy over all prediction probabilities 
    # The entropy score is inversed to sort the list later in ascending order
    pred_entropy = np.sum(np.multiply(probs,np.log(probs)), axis = -1)
    
    # Entropy over the sequence (with no padding)
    # The value is computed as the average entropy w.r.t. the tokens which are not padding
    sequence_len = np.count_nonzero((X_i[0] != 0), -1)
    sequence_entropy = np.divide((pred_entropy * (X_i[0] != 0)).sum(axis = -1)[1:], sequence_len[1:]) # [1:] because first sequence is empty line

    # Get indices of sorted array
    sequence_entropy_sort_index = np.argsort(sequence_entropy)
    # Add 1 to indices to take first line into account
    add1 = np.vectorize(lambda x: x + 1)
    sequence_entropy_sort_index = add1(sequence_entropy_sort_index)

    # Entropy over the sequence (with padding)
    # The value is computed as the sum of entropies of all tokens in the sequence
    #sequence_entropy = np.sum(pred_entropy, axis = -1)    
    #sequence_entropy_sort_index = np.argsort(sequence_entropy)
        
    # Index 0 in the predictions referes to padding
    ind2labelNew = ind2label[0].copy()
    ind2labelNew.update({0: "null"})

    # Compute the labels for each prediction
    pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
    true_label = [[ind2labelNew[x] for x in b] for b in true_index]

    # Flatten to uniform indexing
    pred_flat = np.ravel(pred_label)
    true_flat = np.ravel(true_label)
    entropy_flat = np.ravel(pred_entropy)

    seq_len = maxWords
    
    if write_to_disk:
        os.makedirs(folder_path+"/"+"query_result", exist_ok=True)
    
    value_to_return = None
    
    # Get least confident index rank
    if mode == "least":
        # Compute lowest rank indexing
        query_index_least_rank = sequence_entropy_sort_index[:numSeqToQuery]
        value_to_return = query_index_least_rank
        filename="least.txt"
      
    # Get most confident index rank
    if mode == "most":
        # Compute lowest and highest rank indexing    
        query_index_most_rank = sequence_entropy_sort_index[max((len(sequence_entropy_sort_index) - numSeqToQuery),0):]
        value_to_return = query_index_most_rank
        filename="most.txt"
        
    # Get random index 
    if mode == "random":
        # Compute random indexing
        query_index_random = np.arange(1, len(sequence_entropy_sort_index) + 1)
        np.random.seed(seed)
        np.random.shuffle(query_index_random)
        query_index_random = query_index_random[:numSeqToQuery]
        value_to_return = query_index_random
        filename="random.txt"
                    
    # Get hybrid index 
    if mode == "hybrid":
        # least/most ratio is hardcoded 
        least_ratio = 1/3
        most_ratio = 1 - least_ratio
        # Compute hybrid indexing
        query_index_least_rank = sequence_entropy_sort_index[:int(numSeqToQuery*(least_ratio))]
        query_index_most_rank = sequence_entropy_sort_index[max((len(sequence_entropy_sort_index) - int(numSeqToQuery*(most_ratio))),0):]
        value_to_return = np.concatenate((query_index_least_rank, query_index_most_rank), axis = -1)
        filename="hybrid.txt"

    # Store sequences in a file
    if write_to_disk:
        query_result_path = folder_path+"/"+"query_result"+"/"+filename
        with open(query_result_path, "w", encoding = "utf-8") as f:
            if verbose:
                f.write("token target predicted entropy\r\r")
            else:
                f.write("-DOCSTART- -X- -X- o\r\r")
            for i in value_to_return: 
                seq = X_w[i]
                seq_offset = i*maxWords + seq_len - len(seq)
                for j,w in enumerate(seq):
                    index = seq_offset + j
                    if verbose:
                        f.write(w + " " + true_flat[index] + " " + pred_flat[index] + " " + str(entropy_flat[index]) + "\r")
                    else:
                        if task == 1:
                            f.write(w + " " + true_flat[index] + " o o\r")
                        elif task == 2:
                            f.write(w + " o " + true_flat[index] + " o\r")
                        elif task == 3:
                            f.write(w + " o o " + true_flat[index] + "\r")
                        else:
                            raise Exception('Bad task given.')
                f.write("\r")  
        

    print("CNN query has terminated!")
    return value_to_return

    
def CNN_ActiveModel(task, X_train_w, X_test_w, X_valid_w, y_train_w, y_test_w, y_valid_w, tag_init_min_th, nbr_iters, 
                    nbr_epochs, query_mode, inc_perc = 0.03, word_embedding_size = 100, 
                    char_embedding_size = 50, pretrained_embedding="", folder_path="active_model", store_models = False):
    
    """
        Active learning platform, which does multiple training cycles.
        As a first step, preprocesses the data to unify digits under the same token and splits train set into labeled/unlabelled dataset.
        Then, for each training cycle:
            - Processes the data to get indices and features for the given iteration.
            - Trains the model with the labeled dataset.
            - Computes queries references from the unlabeled dataset.
            - The queried references are removed from unlabeled dataset and appended to labeled dataset.
        
        :param task: task on which active learning is done, must be one these values: "task1", "task2", "task3".
        :param y_train_w: Data to train the model, in the format of load_data function (in utils.py).
        :param y_train_w: Labels of the data to train the model, in the format of load_data function (in utils.py).
        :param X_test_w: Data to test the model, in the format of load_data function (in utils.py).
        :param y_test_w: Labels of the data to test the model, in the format of load_data function (in utils.py).
        :param X_valid_w: Data to train the model, in the format of load_data function (in utils.py).
        :param y_valid_w: Labels of the data to train the model, in the format of load_data function (in utils.py).
        :param tag_init_min_th: Number of tokens for each label in the first training cycle. See splitTrainData for further information.
        :param nbr_iters: Number of training cycles 
        :param nbr_epochs: Number of epochs for each training cycles. Early stopping is not allowed.
        :param inc_perc: Percentage of sequences in X_train_w added at each iteration. Must be between 0 and 1.
        :param query_mode: How references are queried. Must be "least", "most", "hybrid", or "random".
        :param word_embedding_size: See CNN_model (in models.py).
        :param char_embedding_size: See CNN_model  (in models.py).
        :param pretrained_embedding: See CNN_model  (in models.py).
        :param folder_path: Where results, data and weights of each iteration will be stored.
        :param store_models: Store model weights and training dataset at each iteration
        
        :return List with best F1 score at each training cycle.
        
    """
    
    # Check parameters
    assert(tag_init_min_th > 0)
    assert(nbr_iters > 0)
    assert(inc_perc > 0 and inc_perc <= 1)
    
    if store_models:
        if task.lower() == "task1" or task.lower() == "task2" or task.lower() == "task3":
            pass
        else:
            # Must be a valid task: "task1", "task2", "task3"
            print("Not a valid task.")
            raise AssertionError
    
    os.makedirs(folder_path, exist_ok=True)
    file, stdout_original = setPrintToFile("{0}/log.txt".format(folder_path))
    start_time = time.time()
    
    # STEP 0: PREPROCESS DATA
    print("Dataset creation and preprocessing.")
    
    # Merge digits using a specific token
    digits_word = "$NUM$" 
    X_train_w, X_test_w, X_valid_w = mergeDigits([X_train_w, X_test_w, X_valid_w], digits_word)
    
    # Data split to get labelled and unlabelled data for first iteration
    X_train_w_labelled, y_train_w_labelled, X_train_w_unlabelled, y_train_w_unlabelled = splitTrainData(X_train_w, y_train_w, 
                                                                                                        tag_init_min_th)
                                                                                                        
    # Return values
    f1_scores = []
    
    for n_iter in range(nbr_iters):     
        # Iteration strings
        print("\n --- ITERATION " + str(n_iter) + " ---\n")
        iter_task = "iter_{0}".format(n_iter)
        write_data_path = "{0}/iter_{1}/train_active.txt".format(folder_path, n_iter)
        weights_path = "{0}/iter_{1}/iter_{1}.h5".format(folder_path, n_iter)
        os.makedirs(folder_path+"/"+iter_task, exist_ok=True)
        
        # STEP 1: PROCESS DATA
        print("Dataset processing.")
        
        # Store dataset if requested
        if store_models:
            with open(write_data_path, "w", encoding = "utf-8") as f:
                f.write("-DOCSTART- -X- -X- o\r\r")
                # Store least index rank
                for index, (ref_words, ref_tags) in enumerate(zip(X_train_w_labelled, y_train_w_labelled)): 
                    if index == 0:
                        continue
                    for w, t in zip(ref_words, ref_tags):
                        if task.lower() == "task1":
                            f.write(w + " " + t + " o o\r")
                        elif task.lower() == "task2":
                            f.write(w + " o " + t + " o\r")
                        elif task.lower() == "task3":
                            f.write(w + " o o " + t + " \r")
                        else:
                            raise AssertionError
                    f.write("\r")
      
        # Compute indices for words+labels in the TRAINING data
        print("Word counting")
        ukn_words = "out-of-vocabulary"   # Out-of-vocabulary words entry in the "words to index" dictionary
        word2ind, ind2word = indexData_x(X_train_w_labelled, ukn_words)
        print("Label counting")
        label2ind, ind2label =  indexData_y(y_train_w_labelled)

        # Convert data into indices data
        maxlen  = max([len(xx) for xx in X_train_w_labelled])
        padding_style   = 'pre'  # 'pre' or 'post': Style of the padding, in order to have sequence of the same size

        # X padding
        print("Input")
        X_train   = encodePadData_x(X_train_w_labelled,  word2ind,   maxlen, ukn_words, padding_style)
        X_test    = encodePadData_x(X_test_w,   word2ind,   maxlen, ukn_words, padding_style)
        X_valid   = encodePadData_x(X_valid_w,  word2ind,   maxlen, ukn_words, padding_style)
        X_unlabelled = encodePadData_x(X_train_w_unlabelled,  word2ind,   maxlen, ukn_words, padding_style)

        # y padding
        print("Labels")
        y_train  = encodePadData_y(y_train_w_labelled, label2ind, maxlen, padding_style)
        y_test   = encodePadData_y(y_test_w,  label2ind, maxlen, padding_style)
        y_valid  = encodePadData_y(y_valid_w, label2ind, maxlen, padding_style)
        y_unlabelled = encodePadData_y(y_train_w_unlabelled, label2ind, maxlen, padding_style)

        # Create the character level data
        print("Characters")
        char2ind, maxWords, maxChar = characterLevelIndex(X_train_w_labelled, digits_word)
        X_train_char = characterLevelData(X_train_w_labelled, char2ind, maxWords, maxChar, digits_word, padding_style)
        X_test_char  = characterLevelData(X_test_w,  char2ind, maxWords, maxChar, digits_word, padding_style)
        X_valid_char = characterLevelData(X_valid_w, char2ind, maxWords, maxChar, digits_word, padding_style)
        X_unlabelled_char = characterLevelData(y_train_w_unlabelled, char2ind, maxWords, maxChar, digits_word, padding_style)
        
        # STEP 2: TRAIN MODEL
        print("Model training.")
        
        # Training parameters
        batch_size = 128

        # Train model
        epoch, precision, recall, f1 = CNN_model(iter_task, True, [X_train, X_train_char], [X_test, X_test_char], word2ind, maxWords,
                                                 [y_train], [y_test], [ind2label], maxChar, char2ind, pretrained_embedding = 
                                                 pretrained_embedding, word_embedding_size = word_embedding_size, 
                                                 char_embedding_size = char_embedding_size, validation=False, nbr_epochs = nbr_epochs, 
                                                 batch_size = batch_size, optimizer='rmsprop', early_stopping_patience=-1, 
                                                 folder_path=folder_path)
        
        f1_scores.append((epoch, f1))
        
        # This was last iteration
        if n_iter == (nbr_iters - 1):
            print("Training finished.")
            break
            
        # There is no more data to label
        if len(X_train_w_unlabelled) == 1 and X_train_w_unlabelled[0] == []:
            print("No more data to add! Training finished at iteration " + str(n_iter))
            break
        
        # STEP 3: SCORE UNLABELLED DATA
        print("Data scoring.")
        
        # Number of entries to retrieve from unlabelled dataset, as a percentage of the whole training set.
        num_labelled = len(X_train_w_labelled) - 1
        num_unlabelled = len(X_train_w_unlabelled) - 1
        toQuery = int(inc_perc * (num_labelled + num_unlabelled))
        
        # Get score over sequence entropy
        to_label_index = CNN_query(iter_task, X_train_w_unlabelled, [X_unlabelled, X_unlabelled_char], y_unlabelled, 
                                   toQuery, query_mode, 
                                   word2ind, maxWords, [ind2label], maxChar, char2ind,
                                   pretrained_embedding = pretrained_embedding, word_embedding_size = word_embedding_size, 
                                   char_embedding_size = char_embedding_size, optimizer='rmsprop', write_to_disk = False, 
                                   folder_path=folder_path, print_to_file = False)
                                   
        # I don't need weights anymore
        # TODO: find a way to avoid writing/deletion on disk
        if not store_models and os.path.isfile(weights_path):
            os.remove(weights_path)
        
        # STEP 4: SPLIT DATA AND APPEND NEW LABELLED DATA
        print("Appending new data to train set.")
        
        to_label_index_set = set(to_label_index)
        to_label_w = []
        to_label_tag = []
        unlabelled_w = []
        unlabelled_tag = []

        # Split data
        for i, seq in enumerate(X_train_w_unlabelled):
            if i in to_label_index_set and i != 0:
                to_label_w.append(X_train_w_unlabelled[i])
                to_label_tag.append(y_train_w_unlabelled[i])
            elif i != 0:
                unlabelled_w.append(X_train_w_unlabelled[i])
                unlabelled_tag.append(y_train_w_unlabelled[i])
                
        X_train_w_labelled = X_train_w_labelled + to_label_w
        y_train_w_labelled = y_train_w_labelled + to_label_tag
        X_train_w_unlabelled = [[]] + unlabelled_w 
        y_train_w_unlabelled = [[]] + unlabelled_tag

        num_labelled = len(X_train_w_labelled) - 1
        num_unlabelled = len(X_train_w_unlabelled) - 1
        print("Labelled data: " + str(num_labelled) + " entries.")
        print("Unlabelled data: " + str(num_unlabelled) + " entries.")
        print("Usage of full train set: ", str(num_labelled / (num_labelled + num_unlabelled)) + " %.")
        
       
    # FINAL STEP: VALIDATION
    print("Validation.")
    
    CNN_model(iter_task, False, [X_train, X_train_char], [X_test, X_test_char], word2ind, maxWords,
              [y_train], [y_test], [ind2label], maxChar, char2ind, word_embedding_size = word_embedding_size, 
              char_embedding_size = char_embedding_size, pretrained_embedding = pretrained_embedding, validation=True, 
              X_valid=[X_valid, X_valid_char], y_valid= [y_valid], folder_path=folder_path, gen_confusion_matrix=True)
    
    end_time = time.time()
    print("\n\nTotal training time: " + str(end_time - start_time) + " s.\n\n")
    print("Best F1 scores: ", f1_scores)
    closePrintToFile(file, stdout_original)
    # Return list with best F1 scores of each iteration
    return f1_scores



def splitTrainData(X_train_w, y_train_w, tag_min_threshold):
    """
        Splits training dataset in two datasets, which we call "labeled" and "unlabeled" (it's just a convention, both datasets are actually labeled)
        It ensures that the labeled dataset contains a minimum number of entries for each label in the full training dataset.
        The function is used as an initialization for the first training iteration of the active learning model.
        
        :param X_train_w: Data to train the model, in the format of load_data function (in utils.py).
        :param y_train_w: Labels of the data to train the model, in the format of load_data function (in utils.py).
        :param tag_min_threshold: Minimum number of tokens for each label in X_train_w.
        
        :return Four lists: labeled data X, labeled data y, unlabeled data X, unlabeled data y.
    """
    
    # dict { tag -> count in dataset }
    tag_count = defaultdict(int)
    # dict { tag -> indices of sequences that contain tag }
    tag_index = defaultdict(set)
    # dict { tag -> number of time it has been encountered }
    tag_added = defaultdict(int)

    # Histogram of tags
    filled_categories = 0
    for index, tag_seq in enumerate(y_train_w):
        if index == 0:
            continue
        for tag in tag_seq:
            tag_count[tag] += 1
            tag_index[tag].add(index)

    # Initialize indices of labelled and unlabelled datasets
    X_train_unlabelled_index = set(range(len(X_train_w)))
    X_train_labelled_index = set()

    # Create labelled and unlabelled datasets
    tag_count_sorted = sorted(tag_count.items(), key=operator.itemgetter(1))
    for (tag, count) in tag_count_sorted:
        for seq_index in tag_index[tag]:
            if tag_added[tag] < tag_min_threshold:
                if seq_index not in X_train_labelled_index:
                    X_train_labelled_index.add(seq_index)
                    X_train_unlabelled_index.remove(seq_index)
                    for curr_tag in y_train_w[seq_index]:
                        tag_added[curr_tag] += 1
            else:
                break

    # Create labelled dataset
    X_train_w_labelled = []   
    y_train_w_labelled = []
    for ref_index in X_train_labelled_index:  
        if ref_index == 0:
            continue
        X_train_w_labelled.append(X_train_w[ref_index])
        y_train_w_labelled.append(y_train_w[ref_index])

    # Create unlabelled dataset
    X_train_w_unlabelled = []
    y_train_w_unlabelled = []
    for ref_index in X_train_unlabelled_index:  
        if ref_index == 0:
            continue
        X_train_w_unlabelled.append(X_train_w[ref_index])
        y_train_w_unlabelled.append(y_train_w[ref_index])

    num_labelled = len(X_train_w_labelled)
    num_unlabelled = len(X_train_w_unlabelled)
    print("The dataset contains " + str(len(tag_count.keys())) + " labels.")
    print("Number of labelled entries: ", num_labelled)
    print("Number of unlabelled entries: ", num_unlabelled)
    print("Usage of full train set: ", num_labelled / (num_labelled + num_unlabelled))

    # Finalize the dataset by prepending an empty line (default convention for datasets)
    X_train_w_labelled = [[]] + X_train_w_labelled
    y_train_w_labelled = [[]] + y_train_w_labelled
    X_train_w_unlabelled = [[]] + X_train_w_unlabelled
    y_train_w_unlabelled = [[]] + y_train_w_unlabelled
    
    return X_train_w_labelled, y_train_w_labelled, X_train_w_unlabelled, y_train_w_unlabelled
        