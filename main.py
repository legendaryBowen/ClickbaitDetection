import tensorflow as tf
from tensorflow import keras
from utils import *


model = tf.keras.models.load_model('clickbait_detection_model')
print(model.summary())

test_string = "US election 2020: What is the presidential transition"
print(predict(test_string))

