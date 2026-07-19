import numpy as np
from game import Game
from network import build_network, encode_state


def test_encode_state_shapes():
    game = Game()
    board, status, legal = encode_state(game, game.current_player)
    assert board.shape == (9, 3, 3, 2)
    assert status.shape == (9, 4)
    assert legal.shape == (9,)
    assert board.dtype == np.float32
    assert status.dtype == np.float32
    assert legal.dtype == np.float32


def test_network_output_shapes_and_ranges():
    model = build_network()
    game = Game()
    board, status, legal = encode_state(game, game.current_player)
    board_b = np.expand_dims(board, 0)
    status_b = np.expand_dims(status, 0)
    legal_b = np.expand_dims(legal, 0)

    value, policy = model([board_b, status_b, legal_b])
    value = value.numpy()
    policy = policy.numpy()

    assert value.shape == (1, 1)
    assert -1.0 <= value[0, 0] <= 1.0
    assert policy.shape == (1, 81)
    assert np.isclose(policy.sum(), 1.0, atol=1e-4)
    assert np.all(policy >= 0)
