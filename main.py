from keras.models import load_model
from utils import *

# real-word title prediction
def predict(model, test_string):
    test_string = cleanText(test_string)
    test = np.array([test_string])
    test_indices = sentences_to_indices(test, word_to_index, max_len = len(test_string.split()))
    y_pred_onehot = model.predict(test_indices)
    y_pred_binary = onehot_to_binary(y_pred_onehot)
    return y_pred_binary == [1]

# load the glove word embedding file
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B/glove.6B.100d.txt')

# load the model
model = load_model('clickbait_detection_model.h5')
#print(model.summary())

test_string = "Here are 10 facts you dont know"
print(predict(model, test_string))

