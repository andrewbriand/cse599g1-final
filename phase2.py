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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

init_model = keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})

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


def play_n_games(models, n):
  boards = []
  for i in range(n):
    boards.append(Board())

  to_play = 0

  moves = []
  for i in range(n):
    moves.append([])

  while None in [x.result for x in boards]:
    include_legal = model_takes_legal(models[to_play])
    network_inputs = [board_to_network_input(x, include_legal) for x in boards]
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
        network_output = np.multiply(boards[i].get_legal(), network_output + 1e-10)
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

def play_game(models):
  board = Board()
  moves = []
  while board.result is None:
    curr_model = models[board.to_play]
    network_input = np.copy(board.get_cells())
    network_input = np.transpose(network_input, (1, 2, 0))
    if board.to_play == 1:
      network_input = np.flip(network_input, axis=2)
    network_input = np.array([network_input])
    network_output = curr_model.predict(network_input)
    move = np.unravel_index(np.random.choice(np.arange(81), p=network_output.flatten()), network_output[0].shape)
    if board.legal[move[0]][move[1]] == 0.0:
      network_output = np.multiply(board.get_legal(), network_output)[0]
      move = np.unravel_index(np.argmax(network_output), network_output.shape)
    board.make_move(*move)
    #try:
    #  board.make_move(*move)
    #except(e):
    #  print(e)
    #  print("illegal move encountered")
    #  return moves, 0.0 if board.to_play == 1 else 1.0
    moves.append(move)
    
  return moves, board.result

num_models = 20
iters_per_model = 10
curr_model = init_model
sub_batch_size = 16
mini_batch_size = 16*10
learning_rate = 0.002
models = [keras.models.load_model(sys.argv[1], custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy})]
model_save_dir = sys.argv[2]


for m in range(num_models):
  for r in range(iters_per_model):
    #exit()
    moves = []
    results = []
    for i in range(mini_batch_size//sub_batch_size):
      idx = random.randrange(0, len(models))
      opponent = models[idx]
      print("Playing opponent " + str(idx))
      num_wins = 0
      num_draws = 0
      ms, rs = play_n_games((curr_model, opponent), sub_batch_size//2)
      num_wins += rs.count(0.0)
      num_draws += rs.count(0.5)
      results += rs
      moves += ms
      ms, rs = play_n_games((opponent, curr_model), sub_batch_size//2)
      results += rs
      moves += ms
      num_wins += rs.count(1.0)
      num_draws += rs.count(0.5)
      assert(len(rs) - rs.count(1.0) - rs.count(0.5) == rs.count(0.0))
      print("Won " + str(num_wins) + " out of " + str(sub_batch_size) + ", " + str(round(100*num_wins/sub_batch_size, 1)) + "%", "draws: " + str(round(100*num_draws/sub_batch_size, 1)) + "%")
    
    # Replay to calculate gradients
    gradient = [tf.zeros(x.shape) for x in curr_model.trainable_weights]
    #gradient = tf.zeros(curr_model.weights.shape)
    print("Replaying...")
    z = []
    network_inputs = []
    flat_moves = []
    for i, game_moves in enumerate(moves):
      result = results[i]
      board = Board()
      z_i = 0.0
      for m in game_moves: 
        if result == 0.5:
          z_i = 0.0
        elif int(result) == board.to_play:
          z_i = 1.0
        else:
          z_i = -1.0
        include_legal = model_takes_legal(curr_model)
        network_inputs.append(board_to_network_input(board, include_legal))
        z.append(z_i)
        flat_moves.append(m)
        board.make_move(*m)
    z = np.array(z)
    network_inputs = np.array(network_inputs)
    z = tf.convert_to_tensor(z, tf.float32)
    network_inputs = tf.convert_to_tensor(network_inputs)
    output_vars = []
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(curr_model.trainable_weights)
      tape.watch(network_inputs)
      network_outputs = curr_model(network_inputs, training=False)
      #print(network_outputs.shape)
      log_outputs = tf.math.log(network_outputs + 1e-10)
      #log_outputs = network_outputs
      for i, m in enumerate(flat_moves):
        output_vars.append(tf.scalar_mul(z[i], log_outputs[i][m[0]][m[1]]))
      output_vars = tf.convert_to_tensor(output_vars)

    #temp_grad = tape.gradient(x, curr_model.trainable_weights)
    #for t in temp_grad:
    #  print(t.shape)
    #exit()
    #print(tape.gradient(network_outputs, curr_model.weights).shape)
    print("Computing gradient...")
    if tf.math.is_nan(output_vars).numpy().any():
      print("output of network had nan value!")
    grad = tape.gradient(output_vars, curr_model.trainable_weights)
    if True in [tf.math.is_nan(x).numpy().any() for x in grad]:
      print("gradient had nan value!")
    print("computed")
      
      #for i, m in enumerate(flat_moves):
      #  #this_grad = tape.gradient(output_vars[i], curr_model.trainable_weights)
      #  this_grad = grads[i]
      #  for j, g in enumerate(this_grad):
      #    g = tf.cast(g, tf.float32)
      #    g = tf.math.scalar_mul(float(z[i]), g)
      #    gradient[j] += g

      #for i in range(len(gradient)):
        #gradient[i] *= (learning_rate / mini_batch_size)
      
    for i in range(len(grad)):
      grad[i] *= (learning_rate / mini_batch_size)

      curr_model.trainable_weights[i].assign_add(grad[i])

  #### Save the current_model, and add it to the pool
  learning_rate *= 0.8
  curr_model.save(os.path.join(model_save_dir, str(len(models) - 1) + ".h5"))
  print("Adding new model")
  models.append(keras.models.load_model(os.path.join(model_save_dir, str(len(models) - 1) + ".h5"), custom_objects={'move_loss': move_loss, 'move_accuracy' : move_accuracy}))
  print("Added")


  

#while board.result is None:
#  network_input = np.copy(board.get_cells())
#  print(network_input.shape)
#  network_input = np.transpose(network_input, (1, 2, 0))
#  if board.to_play == 1:
#    network_input = np.flip(network_input, axis=2)
#  network_input = tf.convert_to_tensor(np.array([network_input]))
#
#  with tf.GradientTape() as tape:
#    tape.watch(network_input)
#
#    network_output = init_model(network_input, training=False)
#    network_output = tf.reduce_sum(network_output)
#  print(tape.gradient(network_output, init_model.weights))
#  exit()
#  #network_output = np.multiply(board.get_legal(), network_output)
#  #move = np.unravel_index(np.argmax(network_output), network_output.shape)
#  #board.make_move(*move)
#  #board.display()

