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

network = keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})

def board_to_network_input(board):
  network_input = np.copy(board.get_cells())
  network_input = np.transpose(network_input, (1, 2, 0))
  if board.to_play == 1:
    network_input = np.flip(network_input, axis=2)
  return network_input

def get_move_from_network(board):
  network_input = np.array([board_to_network_input(board)])
  network_output = network.predict(network_input)[0]
  network_output = np.multiply(board.get_legal(), network_output + 1e-10)
  network_output = network_output / np.sum(network_output)
  move = np.unravel_index(np.random.choice(np.arange(81), p=network_output.flatten()), network_output.shape)
  return move


win_count = 0
draw_count = 0
n = int(sys.argv[3])
for network_side in range(2):
  for i in range(n//2):
    opponent_proc = subprocess.Popen([sys.argv[2]], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    board = Board()
    if network_side == 1:
      opponent_proc.stdin.write("-1 -1\n0\n".encode())
      opponent_proc.stdin.flush()
    while board.result is None:
      if board.to_play == network_side:
        move = get_move_from_network(board)
        board.make_move(*move)
        opponent_proc.stdin.write((str(move[0]) + " " + str(move[1]) + "\n0\n").encode())
        opponent_proc.stdin.flush()
        print(move)
      else:
        move_str = opponent_proc.stdout.readline().decode('utf-8').split()
        print(move_str)
        move_x = int(move_str[0])
        move_y = int(move_str[1])
        board.make_move(move_x, move_y)
      board.display()

    if board.result == 0.5:
      draw_count += 1
      print("Draw") 
    elif int(board.result) == network_side:
      win_count += 1
      print("Win")
    else:
      print("Loss")
      

win_percentage = 100*win_count/n
draw_percentage = 100*draw_count/n
print("Won: " + str(round(win_percentage, 1)) + "% Drew: " + str(round(draw_percentage, 1)) + "%")
