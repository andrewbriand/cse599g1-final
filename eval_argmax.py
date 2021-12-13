import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import sys
import random
from board import *
from multiprocessing import Pool
import os

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

def play_n_games(models, n):
  boards = []
  for i in range(n):
    boards.append(Board())

  to_play = 0

  moves = []
  for i in range(n):
    moves.append([])

  while None in [x.result for x in boards]:
    network_inputs = [board_to_network_input(x) for x in boards]
    #for i in range(n):
    #  network_input = np.copy(boards[i].get_cells())
    #  network_input = np.transpose(network_input, (1, 2, 0))
    #  if boards[i].to_play == 1:
    #    network_input = np.flip(network_input, axis=2)
    #  network_inputs.append(network_input)
    curr_model = models[to_play]
    network_inputs = np.array(network_inputs)
    network_outputs = curr_model.predict(network_inputs)

    for i in range(n):
      if boards[i].result is None:
        network_output = network_outputs[i]
        #softmax_term = np.exp(network_output / 0.67)
        #network_output = softmax_term / np.sum(softmax_term)
        network_output = np.multiply(boards[i].get_legal(), network_output + 1e-10)
        #print(network_output)
        network_output = network_output / np.sum(network_output)
        move = np.unravel_index(np.random.choice(np.arange(81), p=network_output.flatten()), network_output.shape)
        #if boards[i].legal[move[0]][move[1]] == 0.0:
        #  network_output = np.multiply(boards[i].get_legal(), network_output)
        #  network_output = network_output / 
        #  move = np.unravel_index(np.argmax(network_output), network_output.shape)
        boards[i].make_move(*move)
        moves[i].append(move)
    to_play = 1 if to_play == 0 else 0
  return moves, [x.result for x in boards]

def board_to_network_input(board):
  network_input = np.copy(board.get_cells())
  network_input = np.transpose(network_input, (1, 2, 0))
  if board.to_play == 1:
    network_input = np.flip(network_input, axis=2)
  return network_input

model1 = keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})
model2 = keras.models.load_model(sys.argv[2], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})
n = int(sys.argv[3])

wins = 0
draws = 0
ms, rs = play_n_games((model1, model2), n//2)
wins += rs.count(0.0)
draws += rs.count(0.5)
ms, rs = play_n_games((model2, model1), n//2)
wins += rs.count(1.0)
draws += rs.count(0.5)
print("Won " + str(wins) + ", " + str(wins/n) + "%" + ", draw: " + str(draws/n) + "%")



