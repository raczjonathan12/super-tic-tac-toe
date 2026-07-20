import numpy as np
from game import Game
from network import build_network
from mcts import run_mcts, run_mcts_batch, run_mcts_leaf_parallel, get_policy_target, select_action


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


def test_run_mcts_batch_finds_forced_win_for_every_game_in_the_batch():
    model = build_network()
    games_and_actions = [_make_immediate_win_scenario() for _ in range(4)]
    games = [g for g, _ in games_and_actions]
    roots = run_mcts_batch(games, model, num_simulations=200, add_noise=False)

    assert len(roots) == 4
    for root, (_, win_action) in zip(roots, games_and_actions):
        visit_counts = {a: c.visit_count for a, c in root.children.items()}
        best_action = max(visit_counts, key=visit_counts.get)
        assert best_action == win_action, f"expected {win_action}, got {best_action}, visits={visit_counts}"


def test_run_mcts_batch_finds_forced_block_for_every_game_in_the_batch():
    model = build_network()
    games_and_actions = [_make_forced_block_scenario() for _ in range(4)]
    games = [g for g, _ in games_and_actions]
    roots = run_mcts_batch(games, model, num_simulations=200, add_noise=False)

    for root, (_, block_action) in zip(roots, games_and_actions):
        visit_counts = {a: c.visit_count for a, c in root.children.items()}
        best_action = max(visit_counts, key=visit_counts.get)
        assert best_action == block_action, f"expected {block_action}, got {best_action}, visits={visit_counts}"


def test_run_mcts_batch_makes_far_fewer_model_calls_than_per_game_mcts():
    model = build_network()
    call_count = {"n": 0}
    real_call = model.__call__

    def counting_call(*args, **kwargs):
        call_count["n"] += 1
        return real_call(*args, **kwargs)

    model.__call__ = counting_call

    games = [_make_immediate_win_scenario()[0] for _ in range(4)]
    num_simulations = 50
    run_mcts_batch(games, model, num_simulations=num_simulations, add_noise=False)

    # One batched call for the initial root expansion, plus at most one
    # batched call per simulation round (fewer if some games hit terminal
    # leaves and need no network call that round) — never one per game.
    assert call_count["n"] <= num_simulations + 1
    assert call_count["n"] < 4 * (num_simulations + 1)


def test_run_mcts_leaf_parallel_finds_forced_win():
    game, win_action = _make_immediate_win_scenario()
    model = build_network()
    root = run_mcts_leaf_parallel(game, model, num_simulations=200, leaf_batch_size=8, add_noise=False)

    visit_counts = {a: c.visit_count for a, c in root.children.items()}
    best_action = max(visit_counts, key=visit_counts.get)
    assert best_action == win_action, f"expected {win_action}, got {best_action}, visits={visit_counts}"


def test_run_mcts_leaf_parallel_finds_forced_block():
    game, block_action = _make_forced_block_scenario()
    model = build_network()
    root = run_mcts_leaf_parallel(game, model, num_simulations=200, leaf_batch_size=8, add_noise=False)

    visit_counts = {a: c.visit_count for a, c in root.children.items()}
    best_action = max(visit_counts, key=visit_counts.get)
    assert best_action == block_action, f"expected {block_action}, got {best_action}, visits={visit_counts}"


def test_run_mcts_leaf_parallel_total_visits_match_num_simulations():
    game, _ = _make_immediate_win_scenario()
    model = build_network()
    num_simulations = 160
    root = run_mcts_leaf_parallel(game, model, num_simulations=num_simulations, leaf_batch_size=8, add_noise=False)

    total_visits = sum(c.visit_count for c in root.children.values())
    assert total_visits == num_simulations


def test_run_mcts_leaf_parallel_makes_far_fewer_calls_than_num_simulations():
    model = build_network()
    call_count = {"n": 0}
    real_call = model.__call__

    def counting_call(*args, **kwargs):
        call_count["n"] += 1
        return real_call(*args, **kwargs)

    model.__call__ = counting_call

    game, _ = _make_immediate_win_scenario()
    num_simulations = 160
    leaf_batch_size = 8
    run_mcts_leaf_parallel(game, model, num_simulations=num_simulations,
                            leaf_batch_size=leaf_batch_size, add_noise=False)

    # one batched call per round of leaf_batch_size simulations, plus the
    # initial root expansion -- never one call per simulation.
    max_expected_calls = (num_simulations // leaf_batch_size) + 1
    assert call_count["n"] <= max_expected_calls
    assert call_count["n"] < num_simulations


def test_run_mcts_leaf_parallel_is_faster_than_run_mcts():
    import time

    # A generic opening position, not a forced-tactic scenario: the
    # immediate-win/block scenarios above are too easy for a fair speed
    # comparison, since once run_mcts (plain PUCT) discovers the win,
    # revisiting that same terminal node costs zero network calls, making
    # most of its "simulations" free. That's correct behavior, but it makes
    # the trivial scenario a bad fit for measuring speed. A fresh board has
    # no such shortcut, so both functions must do genuinely comparable
    # amounts of real tree expansion.
    model = build_network()
    game = Game()
    num_simulations = 300

    # Warm up both functions first so neither pays a one-time TF graph
    # tracing cost for a batch size it's never called with before -- that
    # setup cost would otherwise dominate a run this small and make
    # whichever function runs second look artificially slower.
    run_mcts(game, model, num_simulations=10, add_noise=False)
    run_mcts_leaf_parallel(game, model, num_simulations=10, leaf_batch_size=8, add_noise=False)

    start = time.time()
    run_mcts(game, model, num_simulations=num_simulations, add_noise=False)
    sequential_time = time.time() - start

    start = time.time()
    run_mcts_leaf_parallel(game, model, num_simulations=num_simulations, leaf_batch_size=8, add_noise=False)
    parallel_time = time.time() - start

    assert parallel_time < sequential_time
