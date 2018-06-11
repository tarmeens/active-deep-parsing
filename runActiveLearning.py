"""
    DHLAB - IC - EPFL
    
    Run the script to execute the active learning algorithm with the CNN-CNN-LSTM model.
    
    author: Mattia Martinelli
    date: 08/06/2018
    
"""

# Models and Utils scripts
import sys  
sys.path.append("code/")  
from models import *
from utils import *
from active import *

# Get data
X_train_w, y_train1_w, y_train2_w, y_train3_w 	= load_data("dataset/clean_train.txt")	# Training data
X_test_w,  y_test1_w,  y_test2_w,  y_test3_w 	= load_data("dataset/clean_test.txt")	# Testing data
X_valid_w, y_valid1_w, y_valid2_w, y_valid3_w 	= load_data("dataset/clean_valid.txt")	# Validation data

folder_path = "active_model"
tag_init_min_th = 200
nbr_iters = 15
nbr_epochs = 15

# In this example, the active learning algorithm runs on Task 1 and uses least confident (i.e. highest entropy) as an uncertainty sampling measure.
# Change the two parametes below to modify the behaviour of the algorithm.
task = "task1"   # Must be "task1", "task2", or "task3"
query_mode = "least"    # Must be "least", "most", "random", or "hybrid"
CNN_ActiveModel(task, X_train_w, X_test_w, X_valid_w, y_train1_w, y_test1_w, y_valid1_w, tag_init_min_th, nbr_iters, 
                    nbr_epochs, query_mode, inc_perc = 0.03, folder_path = folder_path, pretrained_embedding=True, word_embedding_size=300, char_embedding_size=100, store_models = True)
