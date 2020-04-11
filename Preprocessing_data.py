import nltk
import re
import string
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer #is based on The Porter Stemming Algorithm
from autocorrect import Speller
import pandas as pd
import numpy as np
from link_sheets import df


# def replace_emoji(string):
#     emoji_pattern = re.compile("["
#                            u"\U0001F600-\U0001F64F"  # emotions
#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
#                            "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', string)

# Defining stopwords for this task
stopword = nltk.corpus.stopwords.words("english")
not_stopwords = {'I',  'i', 'am','my', 'myself', 'me','mine',"i'm","I'm"
                     'you','your','yours','yourself','yourselves',
                     'he','his','him','himself',
                     'she','her','hers','herself',
                     'we', 'our', 'ours', 'ourselves',
                     'they','their','themselves','them','theirs',
                     'not'}  # removing some stopwords related to self-disclosure in nltk stopwords
final_stop_words = set([word for word in stopword if word not in not_stopwords])

def cleaning_func():

    # The dataset that is used for cleaning
    dataset = df()


    # Generating the set of punctuations we want to remove
    remove = string.punctuation
    remove = remove.replace("!", "")
    remove = remove.replace(".", "")
    remove = remove.replace("?", "")


    # Defining a speller
    speller = Speller()

    for i in range(len(dataset['Comments'])):

        # 1- removing \n and \t and turning all the characters to lower case
        dataset['Comments'][i] = dataset['Comments'][i].replace('\n', '').replace('\t', '').lower()

        # 2- removing punctuations-{.?!}
        dataset['Comments'][i] = dataset['Comments'][i].translate({ord(char): None for char in remove})

        # 3- Applying stop words and stemming
        dataset['Comments'][i] = ' '.join(word for word in dataset['Comments'][i].split() if word not in final_stop_words)

        # 4- Spelling correction
        dataset['Comments'][i] = ' '.join(speller(word) for word in dataset['Comments'][i].split())

        # replace emojis
        #dataset['Comments'][i] = replace_emoji(dataset['Comments'][i])

        # removing whitespaces
        # dataset['Comments'][i] = dataset['Comments'][i].strip()

    return dataset


