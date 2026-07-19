import numpy as np
from game import Game
from network import build_network
from mcts import run_mcts, get_policy_target, select_action


def _make_immediate_win_scenario():
    # player1 already owns sub-boards 1 and 2 (meta line [0,1,2]); sub-board 0
    # is one move from being won too, which completes the whole game.
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
    game.sub_boards_status[1, 1] = 1
    game.sub_boards_status[1, 0] = 0
    game.sub_boards_status[2, 1] = 1
    game.sub_boards_status[2, 0] = 0
    game.sub_boards_legal[:] = 1
    game.legal_moves()
    return game, 0 * 9 + 2


def _make_forced_block_scenario():
    # player2 already owns sub-boards 1 and 2 (meta line [0,1,2]); sub-board 0
    # is open with player2 two cells into completing it (and the whole game).
    # Every other sub-board is closed (owned by player2 or drawn), so any
    # non-blocking move in sub-board 0 sends the turn right back there.
    game = Game()
    game.current_player = 1
    game.board[:, :, :] = 0
    game.board[:, :, 0] = 1
    game.sub_boards_status[:, :] = 0
    game.sub_boards_status[0, 0] = 1
    game.sub_boards_status[1, 2] = 1
    game.sub_boards_status[2, 2] = 1
    game.sub_boards_status[3:, 3] = 1
    game.board[0, 0, 2] = 1
    game.board[0, 0, 0] = 0
    game.board[0, 1, 2] = 1
    game.board[0, 1, 0] = 0
    game.sub_boards_legal[:] = 0
    game.sub_boards_legal[0] = 1
    game.legal_moves()
    return game, 0 * 9 + 2


def test_mcts_finds_forced_win():
    game, win_action = _make_immediate_win_scenario()
    model = build_network()
    root = run_mcts(game, model, num_simulations=200, add_noise=False)

    visit_counts = {a: c.visit_count for a, c in root.children.items()}
    best_action = max(visit_counts, key=visit_counts.get)
    assert best_action == win_action, f"expected {win_action}, got {best_action}, visits={visit_counts}"


def test_mcts_finds_forced_block_in_narrow_position():
    game, block_action = _make_forced_block_scenario()
    model = build_network()
    root = run_mcts(game, model, num_simulations=200, add_noise=False)

    visit_counts = {a: c.visit_count for a, c in root.children.items()}
    best_action = max(visit_counts, key=visit_counts.get)
    assert best_action == block_action, f"expected {block_action}, got {best_action}, visits={visit_counts}"


def test_policy_target_sums_to_one_and_matches_visits():
    game, _ = _make_immediate_win_scenario()
    model = build_network()
    root = run_mcts(game, model, num_simulations=50, add_noise=False)
    target = get_policy_target(root)
    assert target.shape == (81,)
    assert np.isclose(target.sum(), 1.0, atol=1e-4)
    total_visits = sum(c.visit_count for c in root.children.values())
    for action, child in root.children.items():
        assert np.isclose(target[action], child.visit_count / total_visits)


def test_select_action_zero_temperature_is_argmax_visits():
    game, _ = _make_immediate_win_scenario()
    model = build_network()
    root = run_mcts(game, model, num_simulations=50, add_noise=False)
    action = select_action(root, temperature=1e-3)
    visit_counts = {a: c.visit_count for a, c in root.children.items()}
    assert action == max(visit_counts, key=visit_counts.get)
