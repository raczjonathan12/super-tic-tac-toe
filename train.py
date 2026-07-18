from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import random
from collections import deque

WIN_LINES = np.array([[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[8,4,2]], dtype=int)
class Game():
    def __init__(self):
        board = np.zeros((9,9,3), dtype=int)
        board[:, :, 0] = 1
        self.board = board
        self.sub_boards_status = np.zeros((9, 4), dtype=int)
        self.sub_boards_status[:, 0] = 1
        self.sub_boards_legal = np.ones((9,), dtype=int)
        self.current_player = random.randint(1,2)
        self.game_over = False
        self.winner = None
        self.legal_moves()
    def legal_moves(self):
        is_open = self.sub_boards_status[:, 0]
        moves = is_open & self.sub_boards_legal
        
        cells = self.board[:, :, 0]
        legal_mask = moves[:, np.newaxis] & cells
        self.legal_coords = np.nonzero(legal_mask)
    def check_win(self, current_sub_board=None, is_meta=False): #integer 0-8
        if is_meta:
            meta_board = self.sub_boards_status[:, self.current_player]
            lines = meta_board[WIN_LINES]
            if np.any(np.all(lines == 1, axis=1)):
                self.winner = self.current_player
                self.game_over = True
                return "winner"
            elif not np.any(self.sub_boards_status[:, 0] == 1):
                self.winner = None
                self.game_over = True
                return "draw"
            else:
                return "ongoing"
                     
        else:
            if current_sub_board is not None:
                sub_board = self.board[current_sub_board, :, self.current_player]
                lines = sub_board[WIN_LINES]
                if np.any(np.all(lines == 1, axis=1)):
                    #win to the current player
                    self.sub_boards_status[current_sub_board, self.current_player] = 1
                    self.sub_boards_status[current_sub_board, 0] = 0
                    return "cell_win"
                elif not np.any(np.all(self.board[current_sub_board, :, 0] == 1)):
                    self.sub_boards_status[current_sub_board, 3] = 1
                    self.sub_boards_status[current_sub_board, 0] = 0
                    return "cell_draw"
                else:
                    return "ongoing"

    def set_next_legal(self, played_cell): #index of played cell
        # next_sub_board = self.board[played_cell]
        if self.sub_boards_status[played_cell, 0] == 1:
            self.sub_boards_legal.fill(0)
            self.sub_boards_legal[played_cell] = 1
        else:
            self.sub_boards_legal = self.sub_boards_status[:, 0].copy()
        
    def execute_move(self, sub_board, cell):
        if np.any((self.legal_coords[0] == sub_board) & (self.legal_coords[1] == cell)):
            self.board[sub_board, cell, self.current_player] = 1
            self.board[sub_board, cell, 0] = 0

            win = self.check_win(sub_board)
            if win != "ongoing":
                if win == "cell_win":
                    meta_win = self.check_win(None, is_meta=True)
                    if meta_win != "ongoing":
                        pass
                    else:
                        self.set_next_legal(cell)
                elif win == "cell_draw":
                    self.set_next_legal(cell)
            else:
                self.set_next_legal(cell)
            if not self.game_over:
                self.current_player = 3 - self.current_player
            self.legal_moves()

        else:
            raise ValueError(f"Attempted move {sub_board}, {cell}. Current turn: {self.current_player}")


g = Game()
