import pickle


pickle_in = open("file.pkl", "rb")
embeddings_dictionary = pickle.load(pickle_in)

#embeddings_dictionary is a dictionary of a list with just one element, so we reference it with [0]
print(embeddings_dictionary["me"][0])