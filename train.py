import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import random
from collections import deque
import time
tf.config.list_physical_devices('GPU')
buffer = deque(maxlen=50000)
WIN_LINES = np.array([[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]], dtype=int)
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
                elif not np.any(self.board[current_sub_board, :, 0] == 1):
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
            meta_win = "ongoing"
            win = self.check_win(sub_board)
            if win != "ongoing":
                meta_win = self.check_win(None, is_meta=True)
                self.set_next_legal(cell)
                # if win == "cell_win":
                #     if meta_win != "ongoing":
                #         self.set_next_legal(cell)
                #     else:
                #         self.set_next_legal(cell)
                # elif win == "cell_draw":
                #     self.set_next_legal(cell)
            else:
                self.set_next_legal(cell)
            if not self.game_over:
                self.current_player = 3 - self.current_player
            self.legal_moves()
            return win, meta_win

        else:
            raise ValueError(f"Attempted move {sub_board}, {cell}. Current turn: {self.current_player}")



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
    array = np.stack([mine_board, opponent_board], axis=-1)
    array = array.astype('float32')
    array = np.reshape(array, (9,3,3,2))
    status = np.stack([open_status, mine_status, opponent_status, draw_status], axis=-1)
    status = status.astype('float32')
    legal = game.sub_boards_legal
    legal = legal.astype('float32')
    return array, status, legal


board_input = keras.Input(shape=(9,3,3,2))
status_input = keras.Input(shape=(9,4))
legal_input = keras.Input(shape=(9,))

output_board = layers.TimeDistributed(layers.Flatten())(board_input)
output_board = layers.TimeDistributed(layers.Dense(64, activation='relu'))(output_board)
output_board = layers.Flatten()(output_board)

output_status = layers.Flatten()(status_input)
output_status = layers.Dense(16, activation='relu')(output_status)

output_legal = layers.Dense(12, activation='relu')(legal_input)


output = layers.Concatenate()([output_board, output_status, output_legal])
output = layers.Dense(256, activation='relu')(output)
output = layers.Dense(128, activation='relu')(output)
output = layers.Dense(81, activation=None)(output)

model = keras.Model(inputs=[board_input, status_input, legal_input], outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss='huber'
)
target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

def sync_weights(model, target_model):
    target_model.set_weights(model.get_weights())


def random_legal_action(game):
    random_index = random.randrange(len(game.legal_coords[0]))
    sub_board = game.legal_coords[0][random_index]
    cell = game.legal_coords[1][random_index]
    return sub_board,cell

def legal_mask(game):
    mask = np.full(81, -1e9, dtype=np.float32)
    legal_pair = game.legal_coords
    sub_board = legal_pair[0]
    cell = legal_pair[1]
    flat_index = sub_board * 9 + cell
    mask[flat_index] = 0
    return mask

def greedy_action(game, model):
    board_batch, status_batch, legal_batch = encode_state(game)
    board_batch = np.expand_dims(board_batch, axis=0)
    status_batch = np.expand_dims(status_batch, axis=0)
    legal_batch = np.expand_dims(legal_batch, axis=0)
    pred = model([board_batch, status_batch, legal_batch]).numpy()
    pred = pred[0]
    mask = legal_mask(game)
    masked = pred + mask
    flat_index = np.argmax(masked)
    sub_board = flat_index // 9
    cell = flat_index % 9
    return sub_board, cell

def epsilon_greedy_action(game, model, epsilon):
    if random.random() < epsilon:
        sub_board, cell = random_legal_action(game)
    else:
        sub_board, cell = greedy_action(game, model)
    return sub_board, cell

def subboard_potential(game, player):
    opponent = 3 - player
    mine = np.sum(game.sub_boards_status[:, player] == 1)
    theirs = np.sum(game.sub_boards_status[:, opponent] == 1)
    return mine - theirs

def clone_snapshot(model):
    snapshot = keras.models.clone_model(model)
    snapshot.set_weights(model.get_weights())
    return snapshot

def self_play_batch(epsilon, model, buffer, num_games, opponent_pool=None, pool_fraction=0.3,
                     heuristic_fraction=0.2, shaping_weight=0.05, gamma=0.99):
    games = [Game() for _ in range(num_games)]
    move_counts = [0] * num_games
    active = list(range(num_games))

    pool_opponent = [None] * num_games
    live_player = [None] * num_games
    for i in range(num_games):
        r = random.random()
        if opponent_pool and r < pool_fraction:
            pool_opponent[i] = random.choice(opponent_pool)
            live_player[i] = random.randint(1, 2)
        elif r < pool_fraction + heuristic_fraction:
            pool_opponent[i] = "heuristic"
            live_player[i] = random.randint(1, 2)

    while active:
        random_idx = []
        greedy_idx = []
        opponent_idx = []
        for i in active:
            if pool_opponent[i] is not None and games[i].current_player != live_player[i]:
                opponent_idx.append(i)
            elif random.random() < epsilon:
                random_idx.append(i)
            else:
                greedy_idx.append(i)

        actions = {}
        for i in random_idx:
            actions[i] = random_legal_action(games[i])
        for i in opponent_idx:
            if pool_opponent[i] == "heuristic":
                actions[i] = heuristic_action(games[i])
            else:
                actions[i] = greedy_action(games[i], pool_opponent[i])

        if greedy_idx:
            boards, statuses, legals = [], [], []
            for i in greedy_idx:
                b, s, l = encode_state(games[i])
                boards.append(b)
                statuses.append(s)
                legals.append(l)
            boards = np.stack(boards, axis=0)
            statuses = np.stack(statuses, axis=0)
            legals = np.stack(legals, axis=0)
            preds = model([boards, statuses, legals]).numpy()
            for j, i in enumerate(greedy_idx):
                mask = legal_mask(games[i])
                masked = preds[j] + mask
                flat_index = np.argmax(masked)
                actions[i] = (flat_index // 9, flat_index % 9)

        finished = []
        for i in active:
            game = games[i]
            current_player = game.current_player
            is_live_move = pool_opponent[i] is None or current_player == live_player[i]
            state = encode_state(game, current_player)
            phi_prev = subboard_potential(game, current_player) if is_live_move else 0
            sub_board, cell = actions[i]
            _, meta_win = game.execute_move(sub_board, cell)
            move_counts[i] += 1
            done = game.game_over
            next_legal_mask = legal_mask(game)
            next_state = encode_state(game, game.current_player)
            if meta_win == "winner":
                reward = 1
            elif meta_win == "draw":
                reward = 0
            else:
                reward = 0
            if is_live_move:
                phi_next = subboard_potential(game, current_player)
                reward += shaping_weight * (gamma * phi_next - phi_prev)
                flat_index = sub_board * 9 + cell
                buffer.append((state, flat_index, reward, next_state, done, next_legal_mask))
            if done or move_counts[i] >= 100:
                if not game.game_over:
                    print("a bug happened, needs fixing")
                finished.append(i)

        active = [i for i in active if i not in finished]

    return sum(move_counts)

def training_step(buffer, batch_size, target_model, model, gamma=0.99):
    sample = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones, next_legal_masks = zip(*sample)
    boards, statuses, legals = zip(*states)
    next_boards, next_statuses, next_legals = zip(*next_states)
    boards = np.stack(boards, axis=0)
    statuses = np.stack(statuses, axis=0)
    legals = np.stack(legals, axis=0)
    next_legal_masks = np.stack(next_legal_masks, axis=0)

    next_boards = np.stack(next_boards, axis=0)
    next_statuses = np.stack(next_statuses, axis=0)
    next_legals = np.stack(next_legals, axis=0)

    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    online_next_pred = model([next_boards, next_statuses, next_legals]).numpy() + next_legal_masks
    best_next_actions = np.argmax(online_next_pred, axis=1)
    target_pred = target_model([next_boards, next_statuses, next_legals]).numpy() + next_legal_masks
    next_q = target_pred[np.arange(batch_size), best_next_actions]
    target = rewards - gamma * next_q * (1 - dones)

    pred = model([boards, statuses, legals]).numpy()

    pred[np.arange(batch_size), actions] = target

    loss = model.train_on_batch([boards, statuses, legals], pred)
    return loss


def training_loop(epsilon, decay, episodes, model, buffer, target_model, batch_size, games_per_round=16,
                   pool_fraction=0.3, heuristic_fraction=0.2, pool_size=5, is_kaggle=True):
    epsilon_floor = 0.01
    training_steps = 0
    loss = None
    start_time = time.time()
    total_games = episodes
    rounds = max(1, total_games // games_per_round)
    checkpoint_every = max(1, rounds // 10)
    games_played = 0
    initial_snapshot = clone_snapshot(model)
    opponent_pool = []
    for round_num in range(rounds):
        if round_num % max(1, rounds // 40) == 0:
            elapsed = time.time() - start_time
            print(f'{games_played}/{total_games} games — {elapsed:.0f}s elapsed — epsilon: {epsilon}')
        moves = self_play_batch(epsilon, model, buffer, games_per_round,
                                 opponent_pool=opponent_pool, pool_fraction=pool_fraction,
                                 heuristic_fraction=heuristic_fraction)
        games_played += games_per_round
        for move in range(moves):
            if len(buffer) >= batch_size:
                loss = training_step(buffer, batch_size, target_model, model)
                training_steps += 1
                if training_steps % 500 == 0:
                    sync_weights(model, target_model)
        if epsilon <= epsilon_floor:
            epsilon = epsilon_floor
        else:
            epsilon = epsilon * (decay ** games_per_round)
        if round_num % checkpoint_every == 0:
            if is_kaggle:
                model.save(f'/kaggle/working/model_ep{games_played}.keras')
            else:
                model.save(f'./checkpoints/model_ep{games_played}.keras')
            if loss is not None:
                print(f'loss: {loss}')
            print('vs random:', evaluate(model, 20))
            print('vs heuristic:', evaluate(model, 20, opponent_action=heuristic_action))
            print('vs initial snapshot:', evaluate(model, 20, opponent_action=lambda g: greedy_action(g, initial_snapshot)))
            opponent_pool.append(clone_snapshot(model))
            if len(opponent_pool) > pool_size:
                opponent_pool.pop(0)



def would_win_subboard(game, sub_board, cell, player):
    cells = game.board[sub_board, :, player].copy()
    cells[cell] = 1
    lines = cells[WIN_LINES]
    return np.any(np.all(lines == 1, axis=1))

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

def evaluate(model, num_games, opponent_action=None):
    if opponent_action is None:
        opponent_action = lambda game: random_legal_action(game)
    results = {'win': 0, 'loss': 0, 'draw': 0, 'incomplete': 0}
    for _ in range(num_games):
        game=Game()
        network_player = random.randint(1, 2)
        move_count = 0
        while game.game_over == False and move_count < 100:

            if network_player == game.current_player:
                sub_board, cell = greedy_action(game, model)
            else:
                sub_board, cell = opponent_action(game)
            game.execute_move(sub_board, cell)
        if game.game_over == True:
            if game.winner == network_player:
                results["win"] = results["win"] + 1
            elif game.winner is None:
                results["draw"] = results["draw"] + 1
            else:
                results["loss"] = results['loss'] + 1
        else:
            results["incomplete"] = results['incomplete'] + 1
    return results
if __name__ == "__main__":

    start = time.time()
    training_loop(epsilon=1.0, decay=0.9943, episodes=1000, model=model, buffer=buffer, target_model=target_model, batch_size=64, games_per_round=32, is_kaggle=True)
    end = time.time()
    elapsed = end - start
    print(elapsed)