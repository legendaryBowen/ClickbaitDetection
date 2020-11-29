import numpy as np
import pandas as pd
import json
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from sklearn import preprocessing
from collections import Counter

# import the label dataset
truth_df = pd.DataFrame(columns=['id','truthMedian','truthClass','truthMean'])
with open('data/truth.jsonl') as data:
    for labelobj in data:
        truth = json.loads(labelobj)
        truthlabel = {'id': truth['id'], 'truthMedian': truth['truthMedian'], 'truthClass': truth['truthClass'], 'truthMean': truth['truthMean']}
        truth_df = truth_df.append(truthlabel, ignore_index = True)
#truth_df.head()

# import the title dataset
instances_df = pd.DataFrame(columns=['id','postText'])
with open('data/instances.jsonl') as data:
	for instanceobj in data:
		instance = json.loads(instanceobj)
		instancerow = {'id': instance['id'], 'postText': instance['postText']}
		instances_df = instances_df.append(instancerow, ignore_index=True)
#instances_df.head()

# Merge the label and tilte dataset based on their 'id'
dataset = instances_df.join(truth_df.set_index('id'), on='id')
# Drop the 'id' column
dataset = dataset.drop(labels='id',axis=1)
for i in range(len(dataset)):
    dataset['postText'].values[i] = dataset['postText'].values[i][0]
# Drop Empty dataset
dataset['postText'].dropna(inplace=True)

# Convert 'truthClass' column to 0 or 1
dataset['truthClass'] = dataset['truthClass'].apply(toBinary)
# Convert truthMedian floating number to integer
dataset['truthMedian'] = dataset['truthMedian'].apply(toInteger)
# Clear the postText by removing all punctuations
dataset['cleanPostText'] = dataset['postText'].apply(cleanText)

# Create 'numOfPunctuation' feature to measure the number of punctuations in the postText
dataset['numOfPunctuation'] =  dataset['postText'].apply(count_punc) - dataset['cleanPostText'].apply(count_punc)
# Drop the postText with more than 15 punctuations
dataset.drop(dataset[dataset['numOfPunctuation']>15].index , inplace = True)
dataset = dataset.reset_index()
# Create 'numOfPunctuationNorm' feature 
numOfPunctuation = dataset[['numOfPunctuation']].values
min_max_scaler = preprocessing.MinMaxScaler()
dataset['numOfPunctuationNorm'] = min_max_scaler.fit_transform(numOfPunctuation)

# Drop 'cleanPostText' columns
dataset = dataset.drop(['postText'],axis=1)
dataset = dataset.rename(columns = {'cleanPostText': 'postText'}, inplace = False)

# Create 'stemmingPostText' by stemming the 'postText'
dataset['stemmingPostText'] = dataset['postText'].apply(stemming)

# Obtaining the 'Clickbait words' by measuring every word's frequency difference between clickbait and non-cliackbait datasets
dataset_cb = dataset[dataset['truthClass'] == 1]
dataset_ncb = dataset[dataset['truthClass'] == 0]
cb_words_tuple = Counter(" ".join(dataset_cb["stemmingPostText"]).split()).most_common(300)
cb_words = [words for (words, count) in cb_words_tuple] 
non_cb_words_tuple = Counter(" ".join(dataset_ncb["stemmingPostText"]).split()).most_common(350)
non_cb_words = [words for (words, count) in non_cb_words_tuple] 
true_cb_words = []
for i in range(len(cb_words)):
    word = cb_words[i]
    if word not in non_cb_words[:50+i] and not word.isnumeric():
        true_cb_words.append(word)

# Create 'clickbaitWords' feature by calculating how many 'clickbait words' they have in every 'postText'
countlist = []
for index, row in dataset.iterrows(): 
    words = row["postText"].split()
    count = 0
    for word in words:
        if word in true_cb_words:
            count += 1 
    countlist.append(count)
dataset['clickbaitWords'] = countlist
numOfCbWords = dataset[['clickbaitWords']].values
# Create 'clickbaitWordsNorm' by normalizing 'clickbaitWords'
dataset['clickbaitWordsNorm'] = min_max_scaler.fit_transform(numOfCbWords)
# dataset.head()

# Create 'numOfNumerics' feature by calculating how many numerics they have in every 'postText'
numberCountlist = []
for index, row in dataset.iterrows(): 
    words = row["postText"].split()
    count = 0
    for word in words:
        if word.isnumeric():
            count += 1 
    numberCountlist.append(count)
dataset['numOfNumerics'] = numberCountlist
# Create 'numOfNumericsNorm' by normalizing 'numOfNumerics'
numOfNumerics = dataset[['numOfNumerics']].values
dataset['numOfNumericsNorm'] = min_max_scaler.fit_transform(numOfNumerics)

# import the glove word embedding file
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B/glove.6B.100d.txt')

# length of titles with longest words
maxLen = get_max_length(dataset)
#print("maxLen: ", maxLen)

# split the dataset to training and testing set
train, test = train_test_split(dataset, test_size=0.2)
X_train, Y_train, Y_train_mean = np.array(train["postText"].tolist()), np.array(train["truthMedian"].tolist()), np.array(train["truthMean"].tolist())
# positive_test = test[test["truthClass"] == 1].sample(n=900)
# negative_test = test[test["truthClass"] == 0].sample(n=900)
# test = pd.concat([negative_test, positive_test]).sample(frac=1)
X_test, Y_test, Y_test_mean = np.array(test["postText"].tolist()), np.array(test["truthClass"].tolist()), np.array(test["truthMean"].tolist())
#print(Y_train.shape)
#print(Y_test.shape)

Indices = sentences_to_indices(X_train,word_to_index, maxLen)
#print("X_Train_indices =\n", Indices.shape)

# LSTM model with 3 hidden LSTM layers
def ClickBait_LSTM(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    # Propagate sentence_indices through your embedding layer
    embeddings = embedding_layer(sentence_indices)   
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    # dropout
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=True)(X)
    # dropout
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # dropout
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with 2 units
    X = Dense(4)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)  
    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X) 
    return model

# create our model 
model = ClickBait_LSTM((maxLen,), word_to_vec_map, word_to_index)
print(model.summary())
# compile our model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# preapre the training data
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 4)
print(X_train_indices.shape)

# Training 
model.fit(X_train_indices, Y_train_oh, epochs = 20, batch_size = 32, shuffle=True)

# Convert onehot to binary 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score, mean_squared_error 
y_train_pred_oh = model.predict(X_train_indices)
y_train_pred_binary = onehot_to_binary(y_train_pred_oh)
Y_train_binary = onehot_to_binary(Y_train_oh)

print("Training Error")
print('Accuracy %s' % accuracy_score(Y_train_binary, y_train_pred_binary))
print('Precision %s' % precision_score(Y_train_binary, y_train_pred_binary))
print('Recall %s' % recall_score(Y_train_binary, y_train_pred_binary))
print('F1 score: %s' % f1_score(Y_train_binary, y_train_pred_binary))
print('MSE %s' % mean_squared_error(Y_train_mean, y_train_pred_binary))

print('***********************')
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
y_pred_onehot = model.predict(X_test_indices)
y_pred_binary = onehot_to_binary(y_pred_onehot)

print("Testing Error")
print('Accuracy %s' % accuracy_score(Y_test, y_pred_binary))
print('Precision %s' % precision_score(Y_test, y_pred_binary))
print('Recall %s' % recall_score(Y_test, y_pred_binary))
print('F1 score: %s' % f1_score(Y_test, y_pred_binary))
print('MSE %s' % mean_squared_error(Y_test_mean, y_pred_binary))

print('***********************')
print('Minimum MSE %s' % mean_squared_error(Y_test_mean, Y_test))

# Save our model
model.save("clickbait_detection_model")

# # Error Analysis
# for i in range(1000):
#     if Y_test[i] - y_pred_binary[i] != 0:
#         print(X_test[i])
#         print("Actual Label",Y_test[i])
#         print("Prediction Lable",y_pred_binary[i])
#         print("Prediction",y_pred_onehot[i])
#         print("-------------")

# def test(test_string):
#     test_string = cleanText(test_string)
#     test = np.array([test_string])
#     test_indices = sentences_to_indices(test, word_to_index, max_len = maxLen)
#     y_pred_onehot = model.predict(test_indices)
#     y_pred_binary = onehot_to_binary(y_pred_onehot)
#     if y_pred_binary == [1]:
#         return True
#     else:
#         return False
