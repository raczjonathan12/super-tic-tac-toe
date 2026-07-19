import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def encode_state(game, perspective_player=None):
    if perspective_player is None:
        perspective_player = game.current_player
    mine_index = perspective_player
    opponent_index = 3 - perspective_player

    mine_board = game.board[:, :, mine_index]
    opponent_board = game.board[:, :, opponent_index]
    mine_status = game.sub_boards_status[:, mine_index]
    opponent_status = game.sub_boards_status[:, opponent_index]
    open_status = game.sub_boards_status[:, 0]
    draw_status = game.sub_boards_status[:, 3]

    array = np.stack([mine_board, opponent_board], axis=-1).astype('float32')
    array = np.reshape(array, (9, 3, 3, 2))
    status = np.stack([open_status, mine_status, opponent_status, draw_status], axis=-1).astype('float32')
    legal = game.sub_boards_legal.astype('float32')
    return array, status, legal


def build_network():
    board_input = keras.Input(shape=(9, 3, 3, 2), name="board")
    status_input = keras.Input(shape=(9, 4), name="status")
    legal_input = keras.Input(shape=(9,), name="legal")

    board_feat = layers.TimeDistributed(layers.Flatten())(board_input)
    board_feat = layers.TimeDistributed(layers.Dense(64, activation="relu"))(board_feat)
    board_feat = layers.Flatten()(board_feat)

    status_feat = layers.Flatten()(status_input)
    status_feat = layers.Dense(16, activation="relu")(status_feat)

    legal_feat = layers.Dense(12, activation="relu")(legal_input)

    trunk = layers.Concatenate()([board_feat, status_feat, legal_feat])
    trunk = layers.Dense(256, activation="relu")(trunk)
    trunk = layers.Dense(128, activation="relu")(trunk)

    value_branch = layers.Dense(64, activation="relu")(trunk)
    value = layers.Dense(1, activation="tanh", name="value")(value_branch)

    policy = layers.Dense(81, activation="softmax", name="policy")(trunk)

    model = keras.Model(inputs=[board_input, status_input, legal_input], outputs=[value, policy])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"value": "mse", "policy": "categorical_crossentropy"},
        loss_weights={"value": 1.0, "policy": 1.0},
    )
    return model
