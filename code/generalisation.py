# -*- coding: utf-8 -*-

"""
    DHLAB - IC - EPFL
    
    This file contains functions for generating a CoNLL dataset from bibtex files.
    
    author: Mattia Martinelli
    date: 08/06/2018

"""

# Modules
import os
import random
import bibtexparser 
import codecs
import latexcodec
import re
import sys
import glob
from pybtex.database import parse_file
import glob
sys.path.append("code/")  
from utils import *
from collections import defaultdict


def bibtexToCoNLL(bibpath, writeToDisk = False, outpath = None, 
                    encoding = "latex", mapping = None, seqStruct = None):
    """
        The function transforms a bibtex file in a simil-CoNLL dataset.
        In detail, it parses a bibtex file and returns lists similar to those returned by load_data (in utils.py).
        The lists can either dumped to txt file or not.
        For further information, please consult the report.
        IMPORTANT NOTE: some bibtex files have characters that are not rendered correctly! Use "utf-8" if "latex" does not work.
        Some files are not compatible with this function, as they contain strings that the parser cannot parse.
        
        The function can receive as parameters:
            - A blacklist of bibtex labels that must not be added to the dataset.
            - A custom mapping from bibtex tokens to CoNLL tokens.
            - A reference structure template that is applied to every bibtex entry to build its CoNLL entry.
        If these parameters are not provided, the default hard-coded ones are used.
        
        :param bibpath: input bibtex file path.
        :param writeToDisk: writes on disk the CoNLL file (only for Task 1).
        :param outpath: if writeToDisk = True, output CoNLL file path.
        :param encoding: bibtex parser encoding, e.g. latex or utf-8.
        :param mapping: dictionary that maps bibtex labels in CoNLL labels.
        :param seqStruct: list that represents the label structure of the output references (ordering matters).
        
        :return Two lists: list of tokens and list of labels.

    """
                  
    # DEFAULT STRUCTURES 
    # Maps entry tags in bibtex to labels in dataset    
    if mapping == None:
        mapping = {"author":"author", "title":"title", "pages":"pagination", "year":"year", "journal": "publisher",
                  "volume":"volume", "number":"tomo", "month":"month"
                 }
    # Contains sequence structure (fixed-length sequences). IMPORTANT: ordering matters!
    if seqStruct == None:
        seqStruct = list(["author", "title", "publisher", "volume", "tomo", "year", "pagination"])
    
    begin_line = '-DOCSTART- -X- -X- o\n\n'
    
    # Open input file and parse bibtex
    with open(bibpath, encoding = encoding) as bibtex_file:
        bib_data = parse_file(bibtex_file, bib_format="bibtex")
        
    X_w_out = []
    y_w_out = []
    
    # Iterate over sequences (bibtex entries)
    for key in bib_data.entries:

        # Maps tags to tokens of current sequence
        seqMap = defaultdict(list)
        
        # Extract authors
        if 'author' in bib_data.entries[key].persons:
            for i, author in enumerate(bib_data.entries[key].persons['author']):
                # Add 'and' if multiple authors
                if i >= 1:
                    seqMap["author"].append("and")
                name = author.first_names + author.middle_names + author.prelast_names + author.last_names
                # Split different part names
                for name_part in name:
                    name_part = name_part.replace("{", "").replace("}","")
                    for name_token in re.split('(\W)', name_part):  
                            # Not null or empty string 
                            name_token = name_token.strip()
                            if name_token != "" and name_token != None and not name_token.isspace():
                                seqMap["author"].append(name_token)
                              
        # Extract other fields        
        for k, v in bib_data.entries[key].fields.items():
            # case URL
            if "http" in v or "www" in v:
                continue
            # can be a line with multiple words
            for word in v.replace('\n', ' ').split(" "):
                tokens = re.split('(\W)', word)
                for token in tokens:
                    token = token.replace("{", "").replace("}","")
                    # remove empty tokens
                    if token == "":
                        continue
                    if k in mapping:
                        label = mapping[k]
                    else:
                        label = "o"
                    # Unknown number -> set to 0 so it will be converted in $NUM$ token when processed
                    if (label == "month" or label == "year" or label == "month" or label == 'pagination' or label == 'volume' or label == 'tomo') \
                    and "?" in token:
                        token = 1 # TODO: replace with $NUM$
                    seqMap[label].append(token)
                            
        # Append sequence 
        sequence_x = []
        sequence_y = []
        for seq_label in seqStruct:
            if seq_label in seqMap:
                for token in seqMap[seq_label]:
                    sequence_x.append(token)
                    sequence_y.append(seq_label)
        X_w_out.append(sequence_x)
        y_w_out.append(sequence_y)
            
    # Add first empty line
    X_w_out = [[]] + X_w_out
    y_w_out = [[]] + y_w_out
    
    if writeToDisk:
        if outpath == None:
            outpath = "out.txt"
        # Generate output
        with open(outpath, 'w', encoding = 'utf-8') as f:
            f.write(begin_line)
            for index, (ref_words, ref_tags) in enumerate(zip(X_w_out, y_w_out)): 
                if index == 0:
                    continue
                for w, t in zip(ref_words, ref_tags):
                    f.write(str(w) + " " + str(t) + " o o\r")
                f.write("\r")
            f.write("\n")
        
    return X_w_out, y_w_out


    
def multiBibtexToCoNLL(bib_folder, outpath, encoding = "latex", mapping = None, seqStruct = None):
    """
        The function maps multiple bibtex files to a single simil-CoNLL dataset.
        IMPORTANT NOTE: some files are not compatible with this function, as they contain strings that the parser cannot parse.
        
        :param bib_folder: folder containg the input bibtex files.
        :param outpath:  output CoNLL file path.
        :param encoding: bibtex parser encoding, e.g. latex or utf-8.
        :param mapping: dictionary that maps bibtex labels in CoNLL labels.
        :param seqStruct: list that represents the label structure of the output references (ordering matters).
        
        :return void
        
    """
    # Load the datasets from input folder
    bib_paths = glob.glob(bib_folder + "/*.bib")
    X_w_out = []
    y_w_out = []
    
    begin_line = '-DOCSTART- -X- -X- o\n\n'
    
    # Get sequences
    for bib_path in bib_paths:
        X_w_in, y_w_in = bibtexToCoNLL(bib_path, outpath = None, writeToDisk = False, 
                                        encoding = encoding, mapping = mapping, seqStruct = seqStruct)
        X_w_out += X_w_in[1:]
        y_w_out += y_w_in[1:]

    X_w_out = [[]] + X_w_out
    y_w_out = [[]] + y_w_out
    
    assert(len(X_w_out) == len(y_w_out))
    print("Total number of sequences in dataset: ", len(X_w_out))

    # Dump dataset
    with open(outpath, "w", encoding = "utf-8") as f:
        f.write("-DOCSTART- -X- -X- o\r\r")
        for index, (ref_words, ref_tags) in enumerate(zip(X_w_out, y_w_out)): 
            if index == 0:
                continue
            for w, t in zip(ref_words, ref_tags):
                f.write(str(w) + " " + str(t) + " o o\r")
            f.write("\r")

            
            
def merge(data_folder, outpath):
    """
        The function merges multiple CoNLL files into a single one.
        
        :param data_folder: folder containg the input CoNLL files.
        :param outpath:  output CoNLL file path.
        
        :return void
        
    """
    
    # Load the datasets from input folder
    data_paths = glob.glob(data_folder + "/*.txt")
    X_w_out = []
    y_w_out = []
    
    begin_line = '-DOCSTART- -X- -X- o\n\n'
    
    for data_path in data_paths:
        X_train_w_in, y_train1_w_in, y_train2_w_in, y_train3_w_in = load_data(data_path, split = True)	# Training data
        X_w_out += X_train_w_in[1:]
        y_w_out += y_train1_w_in[1:]

    X_w_out = [[]] + X_w_out
    y_w_out = [[]] + y_w_out
    
    assert(len(X_w_out) == len(y_w_out))
    print("Total number of sequences in dataset: ", len(X_w_out))

    # Dump dataset
    with open(outpath, "w", encoding = "utf-8") as f:
        f.write("-DOCSTART- -X- -X- o\r\r")
        for index, (ref_words, ref_tags) in enumerate(zip(X_w_out, y_w_out)): 
            if index == 0:
                continue
            for w, t in zip(ref_words, ref_tags):
                f.write(str(w) + " " + str(t) + " o o\r")
            f.write("\r")
            