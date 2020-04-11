import torch
from transformers import *
from keras.preprocessing.sequence import pad_sequences
from link_sheets import df
from Preprocessing_data import cleaning_func




dataset = cleaning_func()
comments = dataset["Comments"].tolist()


model_class = BertModel
model_class, tokenizer_class, pretrained_weights = (BertModel,       BertTokenizer,       'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights, )
linear = torch.nn.Linear(model.config.hidden_size, 1)

tokenized_data = [ (tokenizer.encode(sentence, add_special_tokens=True)) for sentence in comments]


# These lines of code are for detecting the comments that have a length more than the defined constraint (400 tokens)
# for index, data in enumerate(tokenized_data):
#     if len(data) > 400:
#         print(index+2)


# saving the bert word embeddings in the embedding column of the dataset
with torch.no_grad():
    for rowNum, sentence in enumerate(comments):
        tokenized_data= torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
        last_hidden_states = model(tokenized_data)[0]
        dataset['Embedding'].iloc[rowNum]=last_hidden_states


#  Finding the max length of the embedding matrix- 3 dimensions
#  The max length = 232
# max_len = 0
# for i in range(dataset.shape[0]):
#   max_len = max(max_len,dataset['Embedding'].iloc[i].shape[1] )
# print(max_len)

# Padding process 
for i in range (dataset.shape[0]):
  dataset['Embedding'].iloc[i] = pad_sequences(dataset['Embedding'].iloc[i], maxlen = 300, padding= 'post')




