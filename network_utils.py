import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

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

def board_to_network_input(board, include_legal = False):
  network_input = np.copy(board.get_cells())
  network_input = np.transpose(network_input, (1, 2, 0))
  if board.to_play == 1:
    network_input = np.flip(network_input, axis=2)
  if include_legal:
    network_input = np.concatenate((network_input, np.copy(np.reshape(board.get_legal(), (9, 9, 1)))), axis=2)
  return network_input

def model_takes_legal(model):
  return model.inputs[0].shape[3] == 3

def get_move_from_network(board):
  network_input = np.array([board_to_network_input(board)])
  network_output = network.predict(network_input)[0]
  network_output = np.multiply(board.get_legal(), network_output + 1e-10)
  network_output = network_output / np.sum(network_output)
  move = np.unravel_index(np.random.choice(np.arange(81), p=network_output.flatten()), network_output.shape)
  return move
