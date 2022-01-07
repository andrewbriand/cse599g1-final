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
from network_utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})

takes_legal = model_takes_legal(model)

output_filename = sys.argv[2]

n = int(sys.argv[3])

input_boards = []
zs = []
batch_size = 1500

while len(input_boards) < n:
  Us = np.random.randint(81, size=batch_size)
  boards = [Board() for x in range(batch_size)]
  move_num = 0
  board_positions = {}
  while None in [x.result for x in boards]:
    network_inputs = np.array([board_to_network_input(b, takes_legal) for b in boards])
    network_outputs = model.predict(network_inputs)
    for j in range(batch_size):
      if boards[j].result is None:
        if move_num == Us[j]:
          legal = np.copy(boards[j].get_legal())
          unif_dist = legal / np.sum(legal)
          move = np.unravel_index(np.random.choice(np.arange(81), p=unif_dist.flatten()), unif_dist.shape)
          boards[j].make_move(*move)
        else:
          if move_num == Us[j] + 1:
            board_positions[j] = (board_to_network_input(boards[j], True), boards[j].to_play)
          network_output = network_outputs[j]
          network_output = np.multiply(boards[j].get_legal(), network_output + 1e-10)
          # Add noise here???
          network_output = network_output / np.sum(network_output)
          move = np.unravel_index(np.random.choice(np.arange(81), p=network_output.flatten()), network_output.shape)
          boards[j].make_move(*move)
     
    move_num += 1
  #print("batch added " + str(len(board_positions)) + " positions")
  for j, (pos, to_play) in board_positions.items():
    input_boards.append(pos)
    if boards[j].result == 0.5:
      zs.append(0.0)
    elif int(boards[j].result) == to_play:
      zs.append(1.0)
    else:
      zs.append(-1.0)
  print("Total positions: " + str(len(input_boards)))


  zs_checkpoint = np.array(zs)
  input_boards_checkpoint = np.array(input_boards)
  
  with(open(output_filename, "wb+")) as f:
    pickle.dump((input_boards_checkpoint, zs_checkpoint), f)
  
