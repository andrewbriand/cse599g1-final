import numpy as np
import json
import pickle
import sys
import random

num_moves_per_output = 3

games = []
args = list(sys.argv[4:])
random.shuffle(args)
for i, filename in enumerate(args):
  if i % 500 == 499:
    print(str(i) + "/" + str(len(args)))
  boards = np.zeros((0, 9, 9, 2))
  labels = np.zeros((0, 9, 9))
  with open(filename, "r") as f:
    obj = json.load(f)
  curr_board = np.zeros((1, 9, 9, 2))
  for frame in obj["frames"][1:]:
    try:
      coords = frame["stdout"].split()
      x = int(coords[0])
      y = int(coords[1])
    except:
      break
    boards = np.concatenate((boards, curr_board))
    new_label = np.zeros((1, 9, 9))
    new_label[0][x][y] = 1.0
    labels = np.concatenate((labels, new_label))
    curr_board[0][x][y][0] = 1.0
    for x in range(9):
      for y in range(9):
        temp = curr_board[0][x][y][0]
        curr_board[0][x][y][0] = curr_board[0][x][y][1]
        curr_board[0][x][y][1] = temp
  games.append((boards, labels))

train_file = sys.argv[1]
valid_file = sys.argv[2]
test_file = sys.argv[3]

train_split = int(0.8 * len(games))
valid_split = int(0.9 * len(games))

train_games = games[0:train_split]
valid_games = games[train_split:valid_split]
test_games = games[valid_split:]

train_boards = np.zeros((0, 9, 9, 2))
train_labels = np.zeros((0, 9, 9))
print("Constructing train set")
for i, g in enumerate(train_games):
  if i % 500 == 499:
    print(str(i) + "/" + str(len(train_games)))
  train_boards = np.concatenate((train_boards, g[0]))
  train_labels = np.concatenate((train_labels, g[1]))

print("train set size: " + str(train_boards.shape[0]))
with open(train_file, "wb+") as f:
  pickle.dump((train_boards, train_labels), f)

valid_boards = np.zeros((0, 9, 9, 2))
valid_labels = np.zeros((0, 9, 9))
for g in valid_games:
  valid_boards = np.concatenate((valid_boards, g[0]))
  valid_labels = np.concatenate((valid_labels, g[1]))

print("valid set size: " + str(valid_boards.shape[0]))
with open(valid_file, "wb+") as f:
  pickle.dump((valid_boards, valid_labels), f)

test_boards = np.zeros((0, 9, 9, 2))
test_labels = np.zeros((0, 9, 9))
for g in test_games:
  test_boards = np.concatenate((test_boards, g[0]))
  test_labels = np.concatenate((test_labels, g[1]))

print("test set size: " + str(test_boards.shape[0]))
with open(test_file, "wb+") as f:
  pickle.dump((test_boards, test_labels), f)



