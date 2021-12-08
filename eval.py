import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import sys
import random

def move_accuracy(y_true, y_pred):
  y_pred = tf.reshape(y_pred, (-1, 81))
  y_true = tf.reshape(y_true, (-1, 81))
  y_pred_idx = tf.argmax(y_pred, axis=1)
  y_true_idx = tf.argmax(y_true, axis=1)
  comp = tf.math.equal(y_pred_idx, y_true_idx)
  return tf.math.count_nonzero(comp) / tf.size(y_true_idx, out_type=tf.dtypes.int64)

def move_loss(y_true, y_pred):
  y_pred = tf.reshape(y_pred, (-1, 81))
  y_true = tf.reshape(y_true, (-1, 81))
  cce = tf.keras.losses.CategoricalCrossentropy()
  return cce(y_true, y_pred)

model = keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})

print(model.summary())

with open(sys.argv[2], "rb") as f:
  (inputs, outputs) = pickle.load(f)

print(model.evaluate(inputs, outputs, batch_size=32*15))
