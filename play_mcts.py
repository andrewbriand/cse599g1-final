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

#mcts = MCTS(policy_network, value_network, 100)
#board = Board()

win_count = 0
draw_count = 0
n = int(sys.argv[4])
for network_side in range(2):
  for i in range(n//2):
    mcts = MCTS(policy_network, value_network, 250)
    opponent_proc = subprocess.Popen([sys.argv[3]], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    board = Board()
    if network_side == 1:
      opponent_proc.stdin.write("-1 -1\n0\n".encode())
      opponent_proc.stdin.flush()
    while board.result is None:
      if board.to_play == network_side:
        move = mcts.get_move()
        mcts.make_move(*move)
        board.make_move(*move)
        board.display()
        opponent_proc.stdin.write((str(move[0]) + " " + str(move[1]) + "\n0\n").encode())
        opponent_proc.stdin.flush()
        print(move)
      else:
        try:
          move_str = opponent_proc.stdout.readline().decode('utf-8').split()
          print(move_str)
          move_x = int(move_str[0])
          move_y = int(move_str[1])
          board.make_move(move_x, move_y)
          mcts.make_move(move_x, move_y)
        except:
          print([x.decode('utf-8') for x in opponent_proc.stderr.readlines()])
          exit()
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
    

