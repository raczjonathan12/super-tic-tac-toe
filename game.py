import random
import numpy as np

WIN_LINES = np.array([[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]], dtype=int)


class Game:
    def __init__(self):
        board = np.zeros((9, 9, 3), dtype=int)
        board[:, :, 0] = 1
        self.board = board
        self.sub_boards_status = np.zeros((9, 4), dtype=int)
        self.sub_boards_status[:, 0] = 1
        self.sub_boards_legal = np.ones((9,), dtype=int)
        self.current_player = random.randint(1, 2)
        self.game_over = False
        self.winner = None
        self.legal_moves()

    def clone(self):
        new_game = Game.__new__(Game)
        new_game.board = self.board.copy()
        new_game.sub_boards_status = self.sub_boards_status.copy()
        new_game.sub_boards_legal = self.sub_boards_legal.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.legal_coords = self.legal_coords
        return new_game

    def legal_moves(self):
        is_open = self.sub_boards_status[:, 0]
        moves = is_open & self.sub_boards_legal
        cells = self.board[:, :, 0]
        legal_mask = moves[:, np.newaxis] & cells
        self.legal_coords = np.nonzero(legal_mask)

    def check_win(self, current_sub_board=None, is_meta=False):
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
                    self.sub_boards_status[current_sub_board, self.current_player] = 1
                    self.sub_boards_status[current_sub_board, 0] = 0
                    return "cell_win"
                elif not np.any(self.board[current_sub_board, :, 0] == 1):
                    self.sub_boards_status[current_sub_board, 3] = 1
                    self.sub_boards_status[current_sub_board, 0] = 0
                    return "cell_draw"
                else:
                    return "ongoing"

    def set_next_legal(self, played_cell):
        if self.sub_boards_status[played_cell, 0] == 1:
            self.sub_boards_legal.fill(0)
            self.sub_boards_legal[played_cell] = 1
        else:
            self.sub_boards_legal = self.sub_boards_status[:, 0].copy()

    def execute_move(self, sub_board, cell):
        if np.any((self.legal_coords[0] == sub_board) & (self.legal_coords[1] == cell)):
            self.board[sub_board, cell, self.current_player] = 1
            self.board[sub_board, cell, 0] = 0
            meta_win = "ongoing"
            win = self.check_win(sub_board)
            if win != "ongoing":
                meta_win = self.check_win(None, is_meta=True)
                self.set_next_legal(cell)
            else:
                self.set_next_legal(cell)
            if not self.game_over:
                self.current_player = 3 - self.current_player
            self.legal_moves()
            return win, meta_win
        else:
            raise ValueError(f"Attempted move {sub_board}, {cell}. Current turn: {self.current_player}")


def legal_action_mask(game):
    mask = np.zeros(81, dtype=bool)
    sub_boards, cells = game.legal_coords
    mask[sub_boards * 9 + cells] = True
    return mask


def random_legal_action(game):
    idx = random.randrange(len(game.legal_coords[0]))
    return int(game.legal_coords[0][idx]), int(game.legal_coords[1][idx])


def would_win_subboard(game, sub_board, cell, player):
    cells = game.board[sub_board, :, player].copy()
    cells[cell] = 1
    lines = cells[WIN_LINES]
    return bool(np.any(np.all(lines == 1, axis=1)))


def heuristic_action(game):
    legal = list(zip(*game.legal_coords))
    player = game.current_player
    opponent = 3 - player
    for sb, c in legal:
        if would_win_subboard(game, sb, c, player):
            return sb, c
    for sb, c in legal:
        if would_win_subboard(game, sb, c, opponent):
            return sb, c
    return random_legal_action(game)
