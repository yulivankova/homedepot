"""
__file__
    param_config.py
__description__
    This file provides global parameter configuration for the project
__adopted from__
    Chenglong Chen < c.chenglong@gmail.com >
"""

import os
import numpy as np

############
## Config ##
############

class ParamConfig:
    def __init__(self,
                 encoding,
                 remove_html,
                 remove_stop_words,
                 remove_numbers,
                 num_features_per_word,
                 min_word_count):

        ## CV params
        self.n_runs =1
        self.n_folds = 3
        self.stratified_label = "relevance"

        ## path
        self.data_folder = 'data'
        self.output_folder = 'output'
        self.original_train_data_path = "%s/train.csv" %self.data_folder
        self.original_product_descriptions_data_path = "%s/product_descriptions.csv" %self.data_folder
        self.original_attributes_data_path = "%s/attributes.csv" %self.data_folder
        self.original_test_data_path = "%s/test.csv" %self.data_folder

        #reading data
        self.encoding = encoding

        #nlp related
        self.remove_html = remove_html
        self.remove_stop_words = remove_stop_words
        self.remove_numbers = remove_numbers

#Initialize a param config
config = ParamConfig(encoding="latin-1",
                     remove_html=True,
                     remove_stop_words=True,
                     remove_numbers=True,
                     num_features_per_word=300,
                     min_word_count=40)
