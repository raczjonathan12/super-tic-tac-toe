import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import random
from collections import deque
import time
tf.config.list_physical_devices('GPU')
buffer = deque(maxlen=50000)
priority_buffer = deque(maxlen=10000)  # transitions with reward != 0 (wins, shaping events)
win_buffer = deque(maxlen=2000)  # transitions with reward >= WIN_THRESHOLD (actual terminal wins only)
WIN_THRESHOLD = 0.5
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



class BatchGame():
    """Vectorized equivalent of Game: holds N games' state as batched numpy
    arrays and updates all of them per ply with array ops instead of a
    Python loop over per-game objects. Semantics must match Game exactly —
    see scratchpad/validate_batch_game.py."""
    def __init__(self, num_games):
        self.N = num_games
        self.board = np.zeros((num_games, 9, 9, 3), dtype=int)
        self.board[:, :, :, 0] = 1
        self.sub_boards_status = np.zeros((num_games, 9, 4), dtype=int)
        self.sub_boards_status[:, :, 0] = 1
        self.sub_boards_legal = np.ones((num_games, 9), dtype=int)
        self.current_player = np.random.randint(1, 3, size=num_games)
        self.game_over = np.zeros(num_games, dtype=bool)
        self.winner = np.zeros(num_games, dtype=int)  # 0 = none/draw

    def legal_mask_full(self):
        is_open = self.sub_boards_status[:, :, 0]
        moves = is_open & self.sub_boards_legal
        cells = self.board[:, :, :, 0]
        legal = moves[:, :, np.newaxis] & cells
        return legal.reshape(self.N, 81).astype(bool)

    def execute_move(self, idx, actions):
        """idx: (n,) array of game indices to update. actions: (n,) flat 0-80."""
        n = len(idx)
        sub_board = actions // 9
        cell = actions % 9
        player = self.current_player[idx]

        self.board[idx, sub_board, cell, player] = 1
        self.board[idx, sub_board, cell, 0] = 0

        sub_cells = self.board[idx, sub_board, :, player]
        sub_lines = sub_cells[:, WIN_LINES]
        cell_win = np.all(sub_lines == 1, axis=2).any(axis=1)

        sub_open_cells = self.board[idx, sub_board, :, 0]
        sub_full = ~np.any(sub_open_cells == 1, axis=1)
        cell_draw = (~cell_win) & sub_full

        win_idx = idx[cell_win]
        self.sub_boards_status[win_idx, sub_board[cell_win], player[cell_win]] = 1
        self.sub_boards_status[win_idx, sub_board[cell_win], 0] = 0

        draw_idx = idx[cell_draw]
        self.sub_boards_status[draw_idx, sub_board[cell_draw], 3] = 1
        self.sub_boards_status[draw_idx, sub_board[cell_draw], 0] = 0

        resolved = cell_win | cell_draw
        win_str = np.full(n, "ongoing", dtype=object)
        win_str[cell_win] = "cell_win"
        win_str[cell_draw] = "cell_draw"
        meta_str = np.full(n, "ongoing", dtype=object)

        if np.any(resolved):
            resolved_positions = np.nonzero(resolved)[0]
            r_idx = idx[resolved]
            r_player = player[resolved]

            meta_board = self.sub_boards_status[r_idx, :, r_player]
            meta_lines = meta_board[:, WIN_LINES]
            meta_win_r = np.all(meta_lines == 1, axis=2).any(axis=1)

            meta_open = self.sub_boards_status[r_idx, :, 0]
            meta_full = ~np.any(meta_open == 1, axis=1)
            meta_draw_r = (~meta_win_r) & meta_full

            meta_str[resolved_positions[meta_win_r]] = "winner"
            meta_str[resolved_positions[meta_draw_r]] = "draw"

            w_idx = r_idx[meta_win_r]
            self.winner[w_idx] = r_player[meta_win_r]
            self.game_over[w_idx] = True
            d_idx = r_idx[meta_draw_r]
            self.game_over[d_idx] = True

        target_open = self.sub_boards_status[idx, cell, 0]
        one_hot = np.zeros((n, 9), dtype=int)
        one_hot[np.arange(n), cell] = 1
        new_legal = np.where(target_open[:, None].astype(bool), one_hot, self.sub_boards_status[idx][:, :, 0])
        self.sub_boards_legal[idx] = new_legal

        not_over = ~self.game_over[idx]
        flip_idx = idx[not_over]
        self.current_player[flip_idx] = 3 - self.current_player[flip_idx]

        return win_str, meta_str


EYE9 = np.eye(9, dtype=int)

def _would_win_all_actions(pattern):
    """pattern: (n,9,9) subboard x cell occupancy for one player.
    Returns (n,81) bool: would placing at this action complete that subboard's line."""
    n = pattern.shape[0]
    hyp = pattern[:, :, None, :] | EYE9[None, None, :, :]  # (n, sb, candidate_cell, cell)
    lines = hyp[:, :, :, WIN_LINES]                         # (n, sb, candidate_cell, 8, 3)
    win = np.all(lines == 1, axis=4).any(axis=3)            # (n, sb, candidate_cell)
    return win.reshape(n, 81)

def tactical_features_batch(mine_board, opponent_board, legal_bool):
    """win_feat[a]=1 if playing action a now completes mover's subboard line.
    block_feat[a]=1 if playing action a now would have completed the opponent's
    subboard line (i.e. this action denies them that win). Both gated by legality."""
    win_feat = _would_win_all_actions(mine_board).astype('float32') * legal_bool
    block_feat = _would_win_all_actions(opponent_board).astype('float32') * legal_bool
    return np.stack([win_feat, block_feat], axis=-1)

def encode_state_batch(game, idx, perspective_players):
    mine_idx = perspective_players
    opp_idx = 3 - perspective_players

    mine_board = game.board[idx, :, :, mine_idx]
    opponent_board = game.board[idx, :, :, opp_idx]
    mine_status = game.sub_boards_status[idx, :, mine_idx]
    opponent_status = game.sub_boards_status[idx, :, opp_idx]
    open_status = game.sub_boards_status[idx, :, 0]
    draw_status = game.sub_boards_status[idx, :, 3]

    array = np.stack([mine_board, opponent_board], axis=-1).astype('float32')
    array = array.reshape(len(idx), 9, 3, 3, 2)
    status = np.stack([open_status, mine_status, opponent_status, draw_status], axis=-1).astype('float32')
    legal = game.sub_boards_legal[idx].astype('float32')
    legal_bool = game.legal_mask_full()[idx].astype('float32')
    tactical = tactical_features_batch(mine_board, opponent_board, legal_bool)
    return array, status, legal, tactical


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
    legal_bool = (legal_mask(game) == 0).astype('float32')
    tactical = tactical_features_batch(mine_board[None, :, :], opponent_board[None, :, :], legal_bool[None, :])[0]
    return array, status, legal, tactical


board_input = keras.Input(shape=(9,3,3,2))
status_input = keras.Input(shape=(9,4))
legal_input = keras.Input(shape=(9,))
tactical_input = keras.Input(shape=(81,2))

output_board = layers.TimeDistributed(layers.Flatten())(board_input)
output_board = layers.TimeDistributed(layers.Dense(64, activation='relu'))(output_board)
output_board = layers.Flatten()(output_board)

output_status = layers.Flatten()(status_input)
output_status = layers.Dense(16, activation='relu')(output_status)

output_legal = layers.Dense(12, activation='relu')(legal_input)

output_tactical = layers.Flatten()(tactical_input)
output_tactical = layers.Dense(32, activation='relu')(output_tactical)

output = layers.Concatenate()([output_board, output_status, output_legal, output_tactical])
output = layers.Dense(256, activation='relu')(output)
output = layers.Dense(128, activation='relu')(output)
output = layers.Dense(81, activation=None)(output)

model = keras.Model(inputs=[board_input, status_input, legal_input, tactical_input], outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-3, clipnorm=1.0),
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
    board_batch, status_batch, legal_batch, tactical_batch = encode_state(game)
    board_batch = np.expand_dims(board_batch, axis=0)
    status_batch = np.expand_dims(status_batch, axis=0)
    legal_batch = np.expand_dims(legal_batch, axis=0)
    tactical_batch = np.expand_dims(tactical_batch, axis=0)
    pred = model([board_batch, status_batch, legal_batch, tactical_batch]).numpy()
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

def subboard_potential_batch(bg, idx, players):
    opponent = 3 - players
    mine = np.sum(bg.sub_boards_status[idx, :, players] == 1, axis=1)
    theirs = np.sum(bg.sub_boards_status[idx, :, opponent] == 1, axis=1)
    return mine - theirs

def heuristic_action_bg(bg, g):
    mask = bg.legal_mask_full()[g]
    legal_actions = np.nonzero(mask)[0]
    player = bg.current_player[g]
    opponent = 3 - player
    for target in (player, opponent):
        for a in legal_actions:
            sb, c = a // 9, a % 9
            cells = bg.board[g, sb, :, target].copy()
            cells[c] = 1
            if np.any(np.all(cells[WIN_LINES] == 1, axis=1)):
                return a
    return random.choice(legal_actions)

def clone_snapshot(model):
    snapshot = keras.models.clone_model(model)
    snapshot.set_weights(model.get_weights())
    return snapshot

def self_play_batch(epsilon, model, buffer, num_games, opponent_pool=None, pool_fraction=0.3,
                     heuristic_fraction=0.2, shaping_weight=0.05, gamma=0.99, priority_buffer=None, win_buffer=None):
    bg = BatchGame(num_games)
    move_counts = np.zeros(num_games, dtype=int)
    active = np.arange(num_games)

    mode = np.zeros(num_games, dtype=int)  # 0 self-play, 1 pool, 2 heuristic
    opponent_model_idx = np.full(num_games, -1, dtype=int)
    live_player = np.zeros(num_games, dtype=int)
    for i in range(num_games):
        r = random.random()
        if opponent_pool and r < pool_fraction:
            mode[i] = 1
            opponent_model_idx[i] = random.randrange(len(opponent_pool))
            live_player[i] = random.randint(1, 2)
        elif r < pool_fraction + heuristic_fraction:
            mode[i] = 2
            live_player[i] = random.randint(1, 2)

    while len(active) > 0:
        current_player = bg.current_player[active]
        is_opponent_turn = (mode[active] != 0) & (current_player != live_player[active])
        is_live_move = ~is_opponent_turn

        heuristic_local = active[is_opponent_turn & (mode[active] == 2)]
        pool_local = active[is_opponent_turn & (mode[active] == 1)]
        live_local = active[is_live_move]

        actions = np.zeros(num_games, dtype=int)

        for g in heuristic_local:
            actions[g] = heuristic_action_bg(bg, g)

        if len(pool_local) > 0:
            groups = {}
            for g in pool_local:
                groups.setdefault(opponent_model_idx[g], []).append(g)
            legal_full = bg.legal_mask_full()
            for snap_idx, games_g in groups.items():
                games_g = np.array(games_g)
                snap_model = opponent_pool[snap_idx]
                boards, statuses, legals, tacticals = encode_state_batch(bg, games_g, bg.current_player[games_g])
                preds = snap_model([boards, statuses, legals, tacticals]).numpy()
                mask = legal_full[games_g]
                masked = np.where(mask, preds, -1e9)
                actions[games_g] = np.argmax(masked, axis=1)

        if len(live_local) > 0:
            rand_mask = np.random.rand(len(live_local)) < epsilon
            random_games = live_local[rand_mask]
            greedy_games = live_local[~rand_mask]
            legal_full = bg.legal_mask_full()

            if len(random_games) > 0:
                mask = legal_full[random_games]
                noise = np.random.rand(len(random_games), 81)
                scores = np.where(mask, noise, -1.0)
                actions[random_games] = np.argmax(scores, axis=1)

            if len(greedy_games) > 0:
                boards, statuses, legals, tacticals = encode_state_batch(bg, greedy_games, bg.current_player[greedy_games])
                preds = model([boards, statuses, legals, tacticals]).numpy()
                mask = legal_full[greedy_games]
                masked = np.where(mask, preds, -1e9)
                actions[greedy_games] = np.argmax(masked, axis=1)

        mover = bg.current_player[active].copy()
        state_arr, state_status, state_legal, state_tactical = encode_state_batch(bg, active, mover)
        phi_prev = subboard_potential_batch(bg, active, mover)

        action_vec = actions[active]
        _, meta_arr = bg.execute_move(active, action_vec)
        move_counts[active] += 1

        done_arr = bg.game_over[active]
        legal_bool = bg.legal_mask_full()[active]
        next_legal_mask_all = np.where(legal_bool, 0.0, -1e9).astype(np.float32)
        next_perspective = bg.current_player[active]
        next_state_arr, next_state_status, next_state_legal, next_state_tactical = encode_state_batch(bg, active, next_perspective)

        reward = np.zeros(len(active), dtype=np.float32)
        reward[meta_arr == "winner"] = 1.0
        phi_next = subboard_potential_batch(bg, active, mover)
        reward += shaping_weight * (gamma * phi_next - phi_prev)

        for p in range(len(active)):
            if not is_live_move[p]:
                continue
            g = active[p]
            transition = (
                (state_arr[p], state_status[p], state_legal[p], state_tactical[p]),
                int(action_vec[p]),
                float(reward[p]),
                (next_state_arr[p], next_state_status[p], next_state_legal[p], next_state_tactical[p]),
                bool(done_arr[p]),
                next_legal_mask_all[p],
            )
            buffer.append(transition)
            if priority_buffer is not None and abs(reward[p]) > 1e-6:
                priority_buffer.append(transition)
            if win_buffer is not None and reward[p] >= WIN_THRESHOLD:
                win_buffer.append(transition)

        timed_out = (move_counts[active] >= 100) & (~done_arr)
        for g in active[timed_out]:
            print("a bug happened, needs fixing")
        finished_mask = done_arr | timed_out
        active = active[~finished_mask]

    return int(np.sum(move_counts))

def training_step(buffer, batch_size, target_model, model, gamma=0.99, priority_buffer=None, priority_fraction=0.3,
                   win_buffer=None, win_fraction=0.2):
    if win_buffer is not None and len(win_buffer) > 0:
        n_win = min(int(batch_size * win_fraction), len(win_buffer))
    else:
        n_win = 0
    if priority_buffer is not None and len(priority_buffer) > 0:
        n_priority = min(int(batch_size * priority_fraction), len(priority_buffer))
    else:
        n_priority = 0
    n_uniform = batch_size - n_priority - n_win
    sample = random.sample(buffer, n_uniform)
    if n_priority > 0:
        sample += random.sample(priority_buffer, n_priority)
    if n_win > 0:
        sample += random.sample(win_buffer, n_win)
    states, actions, rewards, next_states, dones, next_legal_masks = zip(*sample)
    boards, statuses, legals, tacticals = zip(*states)
    next_boards, next_statuses, next_legals, next_tacticals = zip(*next_states)
    boards = np.stack(boards, axis=0)
    statuses = np.stack(statuses, axis=0)
    legals = np.stack(legals, axis=0)
    tacticals = np.stack(tacticals, axis=0)
    next_legal_masks = np.stack(next_legal_masks, axis=0)

    next_boards = np.stack(next_boards, axis=0)
    next_statuses = np.stack(next_statuses, axis=0)
    next_legals = np.stack(next_legals, axis=0)
    next_tacticals = np.stack(next_tacticals, axis=0)

    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)

    combined_boards = np.concatenate([boards, next_boards], axis=0)
    combined_statuses = np.concatenate([statuses, next_statuses], axis=0)
    combined_legals = np.concatenate([legals, next_legals], axis=0)
    combined_tacticals = np.concatenate([tacticals, next_tacticals], axis=0)
    combined_pred = model([combined_boards, combined_statuses, combined_legals, combined_tacticals]).numpy()
    pred = combined_pred[:batch_size]
    online_next_pred = combined_pred[batch_size:] + next_legal_masks
    best_next_actions = np.argmax(online_next_pred, axis=1)

    target_pred = target_model([next_boards, next_statuses, next_legals, next_tacticals]).numpy() + next_legal_masks
    next_q = target_pred[np.arange(batch_size), best_next_actions]
    target = rewards - gamma * next_q * (1 - dones)

    pred[np.arange(batch_size), actions] = target

    loss = model.train_on_batch([boards, statuses, legals, tacticals], pred)
    return loss


def training_loop(epsilon, decay, episodes, model, buffer, target_model, batch_size, games_per_round=16,
                   pool_fraction=0.3, heuristic_fraction=0.4, pool_size=5, train_decimation=4,
                   priority_buffer=None, priority_fraction=0.3, win_buffer=None, win_fraction=0.2, is_kaggle=True):
    epsilon_floor = 0.01
    training_steps = 0
    loss = None
    start_time = time.time()
    total_games = episodes
    rounds = max(1, total_games // games_per_round)
    checkpoint_every = max(1, rounds // 40)
    games_played = 0
    initial_snapshot = clone_snapshot(model)
    opponent_pool = []
    for round_num in range(rounds):
        if round_num % max(1, rounds // 40) == 0:
            elapsed = time.time() - start_time
            print(f'{games_played}/{total_games} games — {elapsed:.0f}s elapsed — epsilon: {epsilon}')
        moves = self_play_batch(epsilon, model, buffer, games_per_round,
                                 opponent_pool=opponent_pool, pool_fraction=pool_fraction,
                                 heuristic_fraction=heuristic_fraction, priority_buffer=priority_buffer,
                                 win_buffer=win_buffer)
        games_played += games_per_round
        train_calls = max(1, moves // train_decimation)
        for _ in range(train_calls):
            if len(buffer) >= batch_size:
                loss = training_step(buffer, batch_size, target_model, model,
                                      priority_buffer=priority_buffer, priority_fraction=priority_fraction,
                                      win_buffer=win_buffer, win_fraction=win_fraction)
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
            print('vs random:', evaluate_batch(model, 20, opponent='random'))
            print('vs heuristic:', evaluate_batch(model, 20, opponent='heuristic'))
            print('vs initial snapshot:', evaluate_batch(model, 20, opponent=initial_snapshot))
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

def evaluate_batch(model, num_games, opponent='random'):
    """Vectorized equivalent of evaluate(): plays all num_games games in
    lockstep on a BatchGame instead of one Game object at a time.
    opponent: 'random', 'heuristic', or a keras model to play greedily."""
    bg = BatchGame(num_games)
    network_player = np.random.randint(1, 3, size=num_games)
    move_counts = np.zeros(num_games, dtype=int)
    active = np.arange(num_games)

    while len(active) > 0:
        current_player = bg.current_player[active]
        is_network_turn = current_player == network_player[active]
        net_idx = active[is_network_turn]
        opp_idx = active[~is_network_turn]

        actions = np.zeros(num_games, dtype=int)
        legal_full = bg.legal_mask_full()

        if len(net_idx) > 0:
            boards, statuses, legals, tacticals = encode_state_batch(bg, net_idx, bg.current_player[net_idx])
            preds = model([boards, statuses, legals, tacticals]).numpy()
            mask = legal_full[net_idx]
            masked = np.where(mask, preds, -1e9)
            actions[net_idx] = np.argmax(masked, axis=1)

        if len(opp_idx) > 0:
            if opponent == 'random':
                mask = legal_full[opp_idx]
                noise = np.random.rand(len(opp_idx), 81)
                scores = np.where(mask, noise, -1.0)
                actions[opp_idx] = np.argmax(scores, axis=1)
            elif opponent == 'heuristic':
                for g in opp_idx:
                    actions[g] = heuristic_action_bg(bg, g)
            else:
                boards, statuses, legals, tacticals = encode_state_batch(bg, opp_idx, bg.current_player[opp_idx])
                preds = opponent([boards, statuses, legals, tacticals]).numpy()
                mask = legal_full[opp_idx]
                masked = np.where(mask, preds, -1e9)
                actions[opp_idx] = np.argmax(masked, axis=1)

        action_vec = actions[active]
        bg.execute_move(active, action_vec)
        move_counts[active] += 1

        done_arr = bg.game_over[active]
        timed_out = (move_counts[active] >= 100) & (~done_arr)
        active = active[~(done_arr | timed_out)]

    results = {'win': 0, 'loss': 0, 'draw': 0, 'incomplete': 0}
    for g in range(num_games):
        if bg.game_over[g]:
            if bg.winner[g] == network_player[g]:
                results['win'] += 1
            elif bg.winner[g] == 0:
                results['draw'] += 1
            else:
                results['loss'] += 1
        else:
            results['incomplete'] += 1
    return results

if __name__ == "__main__":

    start = time.time()
    training_loop(epsilon=1.0, decay=0.999281, episodes=8000, model=model, buffer=buffer, target_model=target_model,
                  batch_size=256, games_per_round=32, pool_fraction=0.3, heuristic_fraction=0.4,
                  pool_size=8, train_decimation=4, priority_buffer=priority_buffer, priority_fraction=0.3,
                  win_buffer=win_buffer, win_fraction=0.4, is_kaggle=False)
    end = time.time()
    elapsed = end - start
    print(elapsed)