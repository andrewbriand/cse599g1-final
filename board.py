import numpy as np

WIN_TABLE = [
    0xff80808080808080,
    0xfff0aa80faf0aa80,
    0xffcc8080cccc8080,
    0xfffcaa80fefcaa80,
    0xfffaf0f0aaaa8080,
    0xfffafaf0fafaaa80,
    0xfffef0f0eeee8080,
    0xffffffffffffffff
]

class Board:
  def __init__(self, cells=np.zeros((2, 9, 9)), legal=np.ones((9, 9)), to_play=0):
    self.cells = np.zeros((2, 9, 9))
    self.squares = np.zeros((2, 3, 3))
    self.square_mask = np.ones((9, 9))
    self.legal = np.ones((9, 9))
    self.to_play = 0
    self.result = None

  def check_victory(self, square):
    bin_rep = 0
    for x in range(3):
      for y in range(3):
        bin_rep = (bin_rep << 1) + int(square[x][y])
    return WIN_TABLE[bin_rep // 64] & (1 << (bin_rep % 64)) != 0

  def get_cells_in_square(self, player, x, y):
      return self.cells[player][x*3:(x+1)*3,y*3:(y+1)*3]

  def make_move(self, x, y):
    #print(x, y)
    #print(self.legal)
    assert(self.legal[x][y] == 1.0)
    self.cells[self.to_play][x][y] = 1.0
    x_global = x // 3
    y_global = y // 3
    x_local = x % 3
    y_local = y % 3
    # check for local cell victory
    if self.check_victory(self.get_cells_in_square(self.to_play, x_global, y_global)):
      self.squares[self.to_play][x_global][y_global] = 1.0
      self.square_mask[x_global*3:(x_global+1)*3,y_global*3:(y_global+1)*3] = 0.0

    if np.count_nonzero(self.get_cells_in_square(0, x_local, y_local) + self.get_cells_in_square(1, x_local, y_local)) == 9:
      self.square_mask[self.to_play][x_global*3:(x_global+1)*3,y_global*3:(y_global+1)*3] = 0.0

    # check for global victory
    if self.check_victory(self.squares[self.to_play]):
      self.result = self.to_play
      return

    square_occupied = self.squares[0][x_local][y_local] != 0.0 or self.squares[1][x_local][y_local] != 0.0
    square_drawn = np.count_nonzero(self.get_cells_in_square(0, x_local, y_local) + self.get_cells_in_square(1, x_local, y_local)) == 9
    whole_board = square_occupied or square_drawn

    self.legal = np.logical_and(self.square_mask, np.logical_and((self.cells[0] == 0.0), (self.cells[1] == 0.0)).astype(float)).astype(float)
    #print(self.legal)
    if not whole_board:
      local_legal = np.copy(self.legal[x_local*3:(x_local+1)*3,y_local*3:(y_local+1)*3])
      self.legal = np.zeros((9, 9))
      self.legal[x_local*3:(x_local+1)*3,y_local*3:(y_local+1)*3] = local_legal
    #print(self.legal)

    if np.count_nonzero(self.legal) == 0:
      self.result = 0.5

    if self.to_play == 1:
      self.to_play = 0
    else:
      self.to_play = 1

  def get_legal(self):
    return self.legal

  def get_cells(self):
    return self.cells

  def display(self):
    result = "0 1 2   3 4 5   6 7 8\n"
    for y in range(9):
      for x in range(9):
        if self.cells[0][x][y] == 1.0:
          result += "X "
        elif self.cells[1][x][y] == 1.0:
          result += "O "
        else:
          result += "- "
        if x == 2 or x == 5:
          result += "| "
      result += str(y) + "\n"
      if y == 2 or y == 5:
        result += "---------------------\n"
       
    print(result)
