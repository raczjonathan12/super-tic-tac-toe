import random
import numpy as np
from game import Game, legal_action_mask, random_legal_action, would_win_subboard, heuristic_action


def test_fresh_game_has_valid_legal_moves():
    game = Game()
    assert len(game.legal_coords[0]) == 81
    mask = legal_action_mask(game)
    assert mask.sum() == 81


def test_execute_move_updates_board_and_switches_player():
    game = Game()
    game.current_player = 1
    mover = game.current_player
    sub_board, cell = random_legal_action(game)
    win, meta_win = game.execute_move(sub_board, cell)
    assert win in ("ongoing", "cell_win", "cell_draw")
    assert game.board[sub_board, cell, mover] == 1
    assert game.board[sub_board, cell, 0] == 0
    if not game.game_over:
        assert game.current_player == 3 - mover


def test_subboard_win_detection():
    game = Game()
    game.current_player = 1
    game.board[:, :, :] = 0
    game.board[:, :, 0] = 1
    game.board[0, 0, 1] = 1
    game.board[0, 0, 0] = 0
    game.board[0, 1, 1] = 1
    game.board[0, 1, 0] = 0
    game.sub_boards_status[:, :] = 0
    game.sub_boards_status[:, 0] = 1
    game.sub_boards_legal[:] = 1
    game.legal_moves()

    win, meta_win = game.execute_move(0, 2)
    assert win == "cell_win"
    assert game.sub_boards_status[0, 1] == 1


def test_would_win_subboard_detects_completion():
    game = Game()
    game.board[:, :, :] = 0
    game.board[:, :, 0] = 1
    game.board[0, 0, 1] = 1
    game.board[0, 0, 0] = 0
    game.board[0, 1, 1] = 1
    game.board[0, 1, 0] = 0
    assert would_win_subboard(game, 0, 2, 1) is True
    assert would_win_subboard(game, 0, 3, 1) is False


def test_heuristic_action_takes_immediate_win():
    game = Game()
    game.current_player = 1
    game.board[:, :, :] = 0
    game.board[:, :, 0] = 1
    game.board[0, 0, 1] = 1
    game.board[0, 0, 0] = 0
    game.board[0, 1, 1] = 1
    game.board[0, 1, 0] = 0
    game.sub_boards_status[:, :] = 0
    game.sub_boards_status[:, 0] = 1
    game.sub_boards_legal[:] = 1
    game.legal_moves()

    sub_board, cell = heuristic_action(game)
    assert (sub_board, cell) == (0, 2)


def test_clone_is_independent_copy():
    game = Game()
    clone = game.clone()
    sub_board, cell = random_legal_action(game)
    game.execute_move(sub_board, cell)
    assert not np.array_equal(game.board, clone.board)
    assert clone.game_over is False or clone.game_over == (clone.winner is not None)


def test_random_rollouts_always_terminate_within_move_cap():
    random.seed(0)
    for _ in range(30):
        game = Game()
        move_count = 0
        while not game.game_over and move_count < 100:
            sub_board, cell = random_legal_action(game)
            game.execute_move(sub_board, cell)
            move_count += 1
        assert game.game_over, "game did not terminate within 100 moves"
