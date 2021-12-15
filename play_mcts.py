from mcts import *
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import numpy as np
import sys
import random
from board import *
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

policy_network = keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})

value_network = keras.models.load_model(sys.argv[2])

mcts = MCTS(policy_network, value_network, 100)
board = Board()

while board.result is None:
  if board.to_play == 0:
    move = mcts.get_move()
    mcts.make_move(*move)
    board.make_move(*move)
    board.display()
  else:
    while True:
      move_str = input().strip().split()
      try:
        mcts.make_move(int(move_str[0]), int(move_str[1]))
        board.make_move(int(move_str[0]), int(move_str[1]))
        break
      except:
        print('Illegal move, try again')
        continue
    board.display()
    

