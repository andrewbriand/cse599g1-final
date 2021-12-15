import numpy as np
from board import *

def board_to_network_input(board, include_legal = False):
  network_input = np.copy(board.get_cells())
  network_input = np.transpose(network_input, (1, 2, 0))
  if board.to_play == 1:
    network_input = np.flip(network_input, axis=2)
  if include_legal:
    network_input = np.concatenate((network_input, np.copy(np.reshape(board.get_legal(), (9, 9, 1)))), axis=2)
  return network_input

class Node:
  
  def __init__(self, board):
    self.P = 0
    self.N = 0
    self.W = 0
    self.Q = 0
    self.board = board
    self.children = []
    self.move = None


class MCTS:

  def __init__(self, policy_network, value_network, rollouts_per_turn):
    self.board = Board()
    self.policy_network = policy_network
    self.value_network = value_network
    self.c_puct = 5.0
    self.rollouts_per_turn = rollouts_per_turn

  def get_move(self):
    self.tree = Node(self.board)
    for i in range(self.rollouts_per_turn):
      self.rollout(self.tree)
    return max(self.tree.children, key=lambda x: x.N).move
    

  def make_move(self, move_x, move_y):
    self.board.make_move(move_x, move_y)

  def score(self, node):
    return node.Q + self.c_puct * node.P

  def rollout(self, node):
    # this is a leaf
    if len(node.children) == 0:
      if node.board.result is not None:
        if node.board.result == 0.5:
          return 0.0
        elif int(node.board.result) == node.board.to_play:
          return 1.0
        else:
          return -1.0
      else:
        node.W = float(self.value_network.predict(np.array([board_to_network_input(node.board, True)]))[0])
        policy_output = self.policy_network.predict(np.array([board_to_network_input(node.board)]))[0]
        for x in range(9):
          for y in range(9):
            if node.board.legal[x][y] == 1.0:
              new_board = node.board.copy()
              new_board.make_move(x, y)
              new_child = Node(new_board)
              new_child.P = policy_output[x][y]
              new_child.move = (x, y)
              node.children.append(new_child)
        return node.W
    else:
      chosen_child = max(node.children, key=lambda x: self.score(x))
      v = self.rollout(chosen_child)
      node.W += v
      node.N += 1
      node.Q = node.W / node.N
      return v

      
