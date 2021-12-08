import numpy as np

class Board:
  def __init__(cells=np.zeros((2, 9, 9)), legal=np.ones((9, 9)), to_play=0):
    self.cells = cells
    self.squares = np.zeros((2, 3, 3))
    self.legal = legal
    self.to_play = 0
    self.result = None

  def check_victory(square):
    return False

  def get_cells_in_square(player, x, y):
      return self.cells[player][x*3:(x+1)*3,y*3:(y+1)*3]

  def coords(x, y):
    assert(self.legal[x][y] == 1.0)
    self.cells[to_play][x][y] == 1.0
    x_global = x // 3
    y_global = y // 3
    x_local = x % 3
    y_local = y % 3
    # check for local cell victory
    if self.check_victory(self.get_cells_in_square(to_play, x_global, y_global)):
      self.squares[to_play][x_global][y_global] = 1.0

    # check for global victory
    if self.check_victory(self.squares[to_play]):
      self.result = to_play
      return

    square_occupied = self.squares[0][x_local][y_local] != 0.0 or self.squares[1][x_local][y_local] != 0.0
    square_drawn = np.count_nonzero(self.get_cells_in_square(0, x_local, y_local) + self.get_cells_in_square(1, x_local, y_local)) == 9
    whole_board = square_occupied or square_drawn

    self.legal = np.logical_and((self.cells[0] == 0.0), (self.cells[1] == 0.0))
    if not whole_board:
      local_legal = self.legal[x_local*3:(x_local+1)*3,y_local*3:(y_local+1)*3]
      self.legal = np.zeros((9, 9))
      self.legal[x_local*3:(x_local+1)*3,y_local*3:(y_local+1)*3] = local_legal

    if np.count_nonzero(self.legal) == 0:
      result = 0.5

  #def move_vector(vec):
  #  assert(np.count_nonzero(vec) == 1)
  #  assert(np.count_nonzero(np.logical_and(vec, self.legal)) == 1)
    

  def get_legal():
    return self.legal

  def get_cells():
    return self.cells
