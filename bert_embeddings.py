"""
This example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of Comments.
This code has the Copyright below and has some modifications to suit our application
-------------------------------------------------------------------------------
Copyright 2019
Ubiquitous Knowledge Processing (UKP) Lab
Technische Universität Darmstadt
-------------------------------------------------------------------------------

"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
from link_sheets import df


#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout



# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')

#extracting comments column from the dataframe
comments = df()["Comments"]

#converting all comments into a list to give as input for model.encode
comments_list = comments.values.tolist()

comments_embeddings = model.encode(comments_list)

# The result is a list of comment embeddings as numpy arrays
for comment, embedding in zip(comments_list, comments_embeddings):
    print("Sentence:", comment)
    print("Embedding:", embedding)
    print("")