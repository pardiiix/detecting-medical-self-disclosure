"""
This example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of Comments.
This code has the Copyright below for BERT embeddings and has some modifications to suit our application
-------------------------------------------------------------------------------
Copyright 2019
Ubiquitous Knowledge Processing (UKP) Lab
Technische Universität Darmstadt
-------------------------------------------------------------------------------

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#########################################################################
#Create dataframe with BERT word embeddings for each comment
#########################################################################


from sentence_transformers import SentenceTransformer, LoggingHandler
from link_sheets import df as google_sheets_df
import logging

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# Load Sentence model (based on BERT) from URL
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

#extracting tokenized comments from the google sheets dataframe
comments = google_sheets_df()["Comments"]



####################################################
max_sent_len = 200
max_vocab_size = 4000
word_seq = [text_to_word_sequence(comment) for comment in comments]

#Vectorizing a text corpus, turning each text into either a sequence of integers
#(each integer being the index of a token in a dictionary)
tokenizer = Tokenizer(num_words = max_vocab_size)

#Updating the internal vocabulary based on a list of texts up to the max_sent_len.
tokenizer.fit_on_texts([' '.join(seq[:max_sent_len]) for seq in word_seq])

# print("vocab size: ", len(tokenizer.word_index)) #vocab size: 799

#converting sequence of words to sequence of indices
X = tokenizer.texts_to_sequences([' '.join(seq[:max_sent_len]) for seq in word_seq])
X = pad_sequences(X, maxlen = max_sent_len, padding= 'post' , truncating='post')

y = google_sheets_df()["Labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=10, test_size=0.3)


#creating a dictionary for BERT such that embeddings_dictionary[token] = word_embedding_vector
embeddings_dictionary = {}
for comment in word_seq:
	my_list = [[token] for token in comment]
	print(my_list)
	for token in my_list:
		embeddings_dictionary[token[0]] = bert_model.encode(token)