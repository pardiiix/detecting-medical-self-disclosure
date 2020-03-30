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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers


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

compute_df_w_embeddings = True
if compute_df_w_embeddings:
	from sentence_transformers import SentenceTransformer, LoggingHandler
	from link_sheets import df as google_sheets_df
	import logging

	#### Just some code to print debug information to stdout
	np.set_printoptions(threshold=100)
	logging.basicConfig(format='%(asctime)s - %(message)s',
	                    datefmt='%Y-%m-%d %H:%M:%S',
	                    level=logging.INFO,
	                    handlers=[LoggingHandler()])

	df_w_embeddings = google_sheets_df()

	# Load Sentence model (based on BERT) from URL
	bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

	#extracting comments column from the dataframe
	comments = df()["Comments"]

	#converting all comments into a list to give as input for bert_model.encode
	comments_list = comments.values.tolist()

	comments_embeddings = bert_model.encode(comments_list)

	#adding the comments embedding to the dataframe df_w_embeddings
	df_w_embeddings["Embedding"] = comments_embeddings

	# The result is a list of comment embeddings as numpy arrays
	for comment, embedding in zip(comments_list, comments_embeddings):
	    print("Sentence:", comment)
	    print("Embedding:", embedding)
	    print("")

	'''
	To see the complete array of each comment(sentence), uncomment the following lines
	The size of each comment vector is 768
	'''
	# import sys
	# np.set_printoptions(threshold=sys.maxsize)
	#print(bert_model.encode(['hi']))

	df_w_embeddings.to_pickle('df_w_embeddings.pickle')
else:
	df_w_embeddings = pd.read_pickle('df_w_embeddings.pickle')

#########################################################################
#Splitting the dataframe into train/dev/test
#########################################################################

#getting the first 1100 comments since we have only labelled these
X = df_w_embeddings["Embedding"].head(1099)
y = df_w_embeddings["Labels"].head(1099)

X = np.array(X.to_list())
y = np.array(y.to_list())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=10, test_size=0.3)

X_train = X_train[..., None]
X_test = X_test[..., None]
X_val = X_val[..., None]

print('Building the model...')
model = Sequential()
# model.add(Dense(64, input_dim=768))
model.add(LSTM(64, input_shape=(None, 1), name='lstm_layer'))
model.add(Dense(1, name='output_layer'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
print('Done building the model.')

print('Start training...')
history = model.fit(X_train, y_train, batch_size=64, epochs=6, verbose = 1, validation_split =0.2) #verbose =1 : see trainig progress for each epoch
print('Done training.')

print('Start validation...')
loss, accuracy, f1_score, precision, recall = model.evaluate(X_val, y_val, verbose = 1)
print('Done validation.')

# print(score)
print("loss: ", loss)
print("accuracy: ", accuracy)
print("f1_score:", f1_score)
print("precision:", precision)
print("recall:", recall)

# print(history.history)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()