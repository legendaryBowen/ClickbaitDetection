import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk import download
download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.layers.embeddings import Embedding
import string
import re

# Read the glove word embedding file and save it as a dictionary
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

# softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# convert integer to onehot
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

# convert onehot to binary(0 negative, 1 positive)
def onehot_to_binary(data):
    binary = []
    for i in range(len(data)):
        if 2/3*data[i][3] + 1/3*data[i][2] > 2/3*data[i][0] + 1/3*data[i][1]:
        #if data[i][3] + data[i][2] > data[i][0] + data[i][1]:
            binary.append(1)
        else:
            binary.append(0)
    return binary

# prepare the embedding layer
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1        
    # define dimensionality of your GloVe word vectors (= 50)
    emb_dim = word_to_vec_map["happy"].shape[0]      
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros((vocab_len,emb_dim))
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_len,emb_dim,trainable = False)
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) 
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

# split the sentence and convert every word to embedding vectors
def sentences_to_indices(X, word_to_index, max_len):   
    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape 
    X_indices = np.zeros((m,max_len))
    for i in range(m):          
        # Convert the ith training sentence in lower case and split is into words
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in word_to_index.keys():
                X_indices[i, j] = word_to_index[w]
                j = j + 1
    return X_indices

# get the maximum length of a sentence in the dataset
def get_max_length(dataset):
    maxLen = 0
    for i in range(len(dataset)):
        sentence = dataset["postText"][i]
        if len(sentence.split()) > maxLen:
            maxLen = len(sentence.split())
    return maxLen

# stemming the postText
def stemming(postText):
    ps = PorterStemmer()
    sentence = word_tokenize(postText)
    newsentence = []
    for word in sentence:
        newsentence.append(ps.stem(word))
    return ' '.join(newsentence)

# remove all punctuations
def cleanText(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text) 
    return text

# convert the 'no-clickbait' or 'clickbait' to binary indicator
def toBinary(truthClass):
    if truthClass == 'no-clickbait':
        return 0
    else:
        return 1
    
# length of a postText
def count_punc(postText):
    return len(postText)

# Convert truthMedian floating number to integer
def toInteger(truthMedian):
    return round(truthMedian*3)

# real-word title prediction
# def predict(test_string):
#     test_string = cleanText(test_string)
#     test = np.array([test_string])
#     test_indices = sentences_to_indices(test, word_to_index, max_len = maxLen)
#     y_pred_onehot = model.predict(test_indices)
#     y_pred_binary = onehot_to_binary(y_pred_onehot)
#     if y_pred_binary == [1]:
#         return True
#     else:
#         return False

