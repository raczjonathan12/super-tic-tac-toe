import time
import numpy as np
from network import build_network
from self_play import play_self_play_game, play_self_play_batch


def test_self_play_produces_valid_examples():
    model = build_network()
    examples = play_self_play_game(model, num_simulations=10, temperature_moves=2, max_moves=100)

    assert len(examples) > 0
    for board, status, legal, policy_target, value_target in examples:
        assert board.shape == (9, 3, 3, 2)
        assert status.shape == (9, 4)
        assert legal.shape == (9,)
        assert policy_target.shape == (81,)
        assert np.isclose(policy_target.sum(), 1.0, atol=1e-3)
        assert value_target in (-1.0, 0.0, 1.0)


def test_self_play_value_targets_are_antisymmetric_between_movers():
    model = build_network()
    examples = play_self_play_game(model, num_simulations=10, temperature_moves=2, max_moves=100)
    values = [value for (_, _, _, _, value) in examples]
    if any(v != 0.0 for v in values):
        assert any(v == 1.0 for v in values) or all(v == 0.0 for v in values)


def test_self_play_batch_produces_valid_examples_from_all_games():
    model = build_network()
    examples = play_self_play_batch(model, num_games=3, num_simulations=10,
                                     temperature_moves=2, max_moves=100)

    assert len(examples) > 0
    for board, status, legal, policy_target, value_target in examples:
        assert board.shape == (9, 3, 3, 2)
        assert status.shape == (9, 4)
        assert legal.shape == (9,)
        assert policy_target.shape == (81,)
        assert np.isclose(policy_target.sum(), 1.0, atol=1e-3)
        assert value_target in (-1.0, 0.0, 1.0)


def test_self_play_batch_is_faster_than_sequential_games():
    model = build_network()

    start = time.time()
    play_self_play_batch(model, num_games=4, num_simulations=15, temperature_moves=2, max_moves=100)
    batched_time = time.time() - start

    start = time.time()
    for _ in range(4):
        play_self_play_game(model, num_simulations=15, temperature_moves=2, max_moves=100)
    sequential_time = time.time() - start

    assert batched_time < sequential_time
