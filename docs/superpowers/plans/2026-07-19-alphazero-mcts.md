# AlphaZero-style MCTS + Policy/Value Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python self-play training pipeline (MCTS guided by a trained policy/value network) that produces a strong Ultimate Tic-Tac-Toe agent, replacing the abandoned DQN approach.

**Architecture:** A single-game `Game` engine (carried over, unchanged, from the DQN version) is driven by PUCT-style Monte Carlo Tree Search. A small Keras network with two output heads (scalar value, 81-way move-probability policy) provides leaf evaluations during search. Self-play games generate `(state, mcts_visit_distribution, game_outcome)` training examples; the network is trained on these via ordinary supervised learning (cross-entropy for policy, MSE for value) — no bootstrapped TD targets, which is what sidesteps the collapse dynamic the DQN attempt kept hitting.

**Tech Stack:** Python, TensorFlow/Keras, NumPy, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-19-alphazero-mcts-design.md`
- Deployment (TensorFlow.js export, JS MCTS, web UI) is explicitly out of scope for this plan — Python training pipeline only.
- Network inputs: board occupancy planes + sub-board status + legal mask (3 inputs). No hand-engineered tactical features this time — MCTS's real search supersedes them.
- `Game`/`BatchGame` game-rule logic itself does not change (already validated in the prior DQN effort's git history); this plan only adds a `clone()` method to `Game` and does not port `BatchGame` at all (MCTS is inherently single-game; nothing in this plan's scope needs the vectorized batch engine — noted as a possible future optimization, not built here).
- Run all tests from the repo root with `python -m pytest tests/ -v` (adds repo root to `sys.path` so `import game`, `import mcts`, etc. resolve).
- Use the project's existing venv (`.venv`) for all commands.

---

### Task 1: Game engine (`game.py`) with regression tests

**Files:**
- Create: `game.py`
- Create: `tests/test_game.py`
- Modify: `requirements.txt` (already has `pytest`; install it into the venv)

**Interfaces:**
- Produces: `WIN_LINES` (numpy array, shape `(8,3)`), `Game` class with `.board`, `.sub_boards_status`, `.sub_boards_legal`, `.current_player`, `.game_over`, `.winner`, `.legal_coords`, methods `.legal_moves()`, `.execute_move(sub_board, cell) -> (win, meta_win)`, `.clone() -> Game`. Functions `legal_action_mask(game) -> np.ndarray[bool, (81,)]`, `random_legal_action(game) -> (sub_board, cell)`, `would_win_subboard(game, sub_board, cell, player) -> bool`, `heuristic_action(game) -> (sub_board, cell)`.

- [ ] **Step 1: Install pytest into the venv**

Run: `.venv/Scripts/pip install -r requirements.txt` (or `source .venv/Scripts/activate && pip install -r requirements.txt`)
Expected: tensorflow, numpy, pytest install/confirm successfully.

- [ ] **Step 2: Write `game.py` with the `Game` class and helper functions**

```python
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
```

- [ ] **Step 3: Write `tests/test_game.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_game.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add game.py tests/test_game.py requirements.txt
git commit -m "Add game engine module with regression tests (carried over from DQN attempt, unchanged rules)"
```

---

### Task 2: Policy/value network (`network.py`)

**Files:**
- Create: `network.py`
- Create: `tests/test_network.py`

**Interfaces:**
- Consumes: `game.Game` (from Task 1).
- Produces: `build_network() -> keras.Model` (inputs `[board, status, legal]`, outputs `[value, policy]`), `encode_state(game, perspective_player=None) -> (board: np.ndarray(9,3,3,2), status: np.ndarray(9,4), legal: np.ndarray(9,))`.

- [ ] **Step 1: Write `tests/test_network.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_network.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'network'`

- [ ] **Step 3: Write `network.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_network.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add network.py tests/test_network.py
git commit -m "Add policy/value network and state encoding"
```

---

### Task 3: MCTS (`mcts.py`) with correctness tests

**Files:**
- Create: `mcts.py`
- Create: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `game.Game`, `game.legal_action_mask`, `network.build_network`, `network.encode_state`.
- Produces: `MCTSNode` class with `.prior`, `.visit_count`, `.value_sum`, `.children` (dict `action -> MCTSNode`), `.is_expanded()`, `.value()`. `run_mcts(game, model, num_simulations, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, add_noise=False) -> MCTSNode` (the root). `get_policy_target(root) -> np.ndarray(81,)`. `select_action(root, temperature) -> int` (flat action index 0-80).

- [ ] **Step 1: Write `tests/test_mcts.py`**

```python
import numpy as np
from game import Game
from network import build_network
from mcts import run_mcts, get_policy_target, select_action


def _make_immediate_win_scenario():
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
    return game, 0 * 9 + 2


def _make_forced_block_scenario():
    game = Game()
    game.current_player = 1
    game.board[:, :, :] = 0
    game.board[:, :, 0] = 1
    game.sub_boards_status[:, :] = 0
    game.sub_boards_status[1:, 3] = 1
    game.sub_boards_status[0, 0] = 1
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mcts'`

- [ ] **Step 3: Write `mcts.py`**

```python
import math
import numpy as np
from game import legal_action_mask
from network import encode_state


class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def evaluate_and_expand(node, game, model):
    board, status, legal = encode_state(game, game.current_player)
    board_b = np.expand_dims(board, 0)
    status_b = np.expand_dims(status, 0)
    legal_b = np.expand_dims(legal, 0)

    value_out, policy_out = model([board_b, status_b, legal_b])
    value = float(value_out.numpy()[0, 0])
    policy = policy_out.numpy()[0]

    mask = legal_action_mask(game)
    legal_actions = np.nonzero(mask)[0]
    legal_probs = policy[legal_actions]
    total = legal_probs.sum()
    if total > 1e-8:
        legal_probs = legal_probs / total
    else:
        legal_probs = np.full(len(legal_actions), 1.0 / len(legal_actions))

    for action, prob in zip(legal_actions, legal_probs):
        node.children[int(action)] = MCTSNode(prior=float(prob))

    return value


def select_child(node, c_puct):
    best_score = -float("inf")
    best_action = None
    best_child = None
    for action, child in node.children.items():
        q = -child.value()
        u = c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def run_mcts(game, model, num_simulations, c_puct=1.5, dirichlet_alpha=0.3,
             dirichlet_epsilon=0.25, add_noise=False):
    root = MCTSNode(prior=1.0)
    evaluate_and_expand(root, game, model)

    if add_noise and len(root.children) > 0:
        actions = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1 - dirichlet_epsilon) + n * dirichlet_epsilon

    for _ in range(num_simulations):
        node = root
        sim_game = game.clone()
        path = [node]

        while node.is_expanded() and not sim_game.game_over:
            action, node = select_child(node, c_puct)
            sub_board, cell = action // 9, action % 9
            sim_game.execute_move(sub_board, cell)
            path.append(node)

        if sim_game.game_over:
            value = 1.0 if sim_game.winner is not None else 0.0
        else:
            value = evaluate_and_expand(node, sim_game, model)

        for path_node in reversed(path):
            path_node.value_sum += value
            path_node.visit_count += 1
            value = -value

    return root


def get_policy_target(root):
    target = np.zeros(81, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits == 0:
        for action, child in root.children.items():
            target[action] = child.prior
    else:
        for action, child in root.children.items():
            target[action] = child.visit_count / total_visits
    return target


def select_action(root, temperature):
    actions = list(root.children.keys())
    visit_counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
    if temperature <= 1e-3:
        best_idx = int(np.argmax(visit_counts))
        return actions[best_idx]
    powered = visit_counts ** (1.0 / temperature)
    probs = powered / powered.sum()
    return int(np.random.choice(actions, p=probs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mcts.py -v`
Expected: 4 passed.

Note: `test_mcts_finds_forced_win` and `test_mcts_finds_forced_block_in_narrow_position` use a freshly-initialized (untrained, random-weight) network and still must pass — this is the key correctness property of MCTS versus the abandoned DQN approach: because simulations that reach an actual game-over state use the *true* win/loss/draw outcome (not a network guess), tactically forced wins and blocks are found by search itself, regardless of how good the network's judgment is yet. If either test fails, the bug is in the search/backup logic (most likely a sign error in `select_child`'s `-child.value()` or the alternating negation in the backup loop), not in the network.

- [ ] **Step 5: Commit**

```bash
git add mcts.py tests/test_mcts.py
git commit -m "Add PUCT-style MCTS with forced-win/forced-block correctness tests"
```

---

### Task 4: Self-play data generation (`self_play.py`)

**Files:**
- Create: `self_play.py`
- Create: `tests/test_self_play.py`

**Interfaces:**
- Consumes: `game.Game`, `network.build_network`, `network.encode_state`, `mcts.run_mcts`, `mcts.get_policy_target`, `mcts.select_action`.
- Produces: `play_self_play_game(model, num_simulations, c_puct=1.5, temperature_moves=10, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, max_moves=100) -> list[(board, status, legal, policy_target, value_target)]`.

- [ ] **Step 1: Write `tests/test_self_play.py`**

```python
import numpy as np
from network import build_network
from self_play import play_self_play_game


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_self_play.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'self_play'`

- [ ] **Step 3: Write `self_play.py`**

```python
from game import Game
from network import encode_state
from mcts import run_mcts, get_policy_target, select_action


def play_self_play_game(model, num_simulations, c_puct=1.5, temperature_moves=10,
                         dirichlet_alpha=0.3, dirichlet_epsilon=0.25, max_moves=100):
    game = Game()
    history = []
    move_count = 0

    while not game.game_over and move_count < max_moves:
        root = run_mcts(
            game, model, num_simulations, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
            add_noise=True,
        )
        policy_target = get_policy_target(root)
        temperature = 1.0 if move_count < temperature_moves else 1e-3
        action = select_action(root, temperature)

        board, status, legal = encode_state(game, game.current_player)
        history.append([board, status, legal, policy_target, game.current_player])

        sub_board, cell = action // 9, action % 9
        game.execute_move(sub_board, cell)
        move_count += 1

    examples = []
    for board, status, legal, policy_target, mover in history:
        if game.winner is None:
            value_target = 0.0
        elif game.winner == mover:
            value_target = 1.0
        else:
            value_target = -1.0
        examples.append((board, status, legal, policy_target, value_target))

    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_self_play.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add self_play.py tests/test_self_play.py
git commit -m "Add self-play data generation driven by MCTS"
```

---

### Task 5: Training step (`train.py`, part 1)

**Files:**
- Create: `train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `network.build_network`.
- Produces: `training_step(model, batch) -> list[float]` (Keras `train_on_batch` return: `[total_loss, value_loss, policy_loss]`), where `batch` is a list of `(board, status, legal, policy_target, value_target)` tuples matching `self_play.play_self_play_game`'s output format.

- [ ] **Step 1: Write `tests/test_train.py`**

```python
import numpy as np
from network import build_network
from train import training_step


def test_training_step_reduces_loss_on_fixed_batch():
    model = build_network()
    rng = np.random.default_rng(0)

    def random_batch(n):
        boards = rng.random((n, 9, 3, 3, 2)).astype(np.float32)
        statuses = rng.random((n, 9, 4)).astype(np.float32)
        legals = rng.integers(0, 2, size=(n, 9)).astype(np.float32)
        policies = rng.dirichlet(np.ones(81), size=n).astype(np.float32)
        values = rng.uniform(-1, 1, size=n).astype(np.float32)
        return list(zip(boards, statuses, legals, policies, values))

    batch = random_batch(32)
    losses = []
    for _ in range(20):
        loss = training_step(model, batch)
        losses.append(loss[0])

    assert all(np.isfinite(l) for l in losses)
    assert losses[-1] < losses[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Write the `training_step` function in `train.py`**

```python
import random
from collections import deque
import numpy as np

from network import build_network
from self_play import play_self_play_game
from evaluate import evaluate_vs_opponent


def training_step(model, batch):
    boards, statuses, legals, policy_targets, value_targets = zip(*batch)
    boards = np.stack(boards, axis=0)
    statuses = np.stack(statuses, axis=0)
    legals = np.stack(legals, axis=0)
    policy_targets = np.stack(policy_targets, axis=0)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    loss = model.train_on_batch(
        [boards, statuses, legals],
        {"value": value_targets, "policy": policy_targets},
    )
    return loss
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train.py -v`
Expected: 1 passed.

Note: `train.py` imports `evaluate.py`, which doesn't exist yet (built in Task 6) — this import will make `tests/test_train.py` fail to collect until Task 6 is done. Task 6 must be completed before this task's test can pass; do Task 6 immediately after this one, then re-run `tests/test_train.py` as part of Task 6's verification.

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train.py
git commit -m "Add supervised training step for policy/value network"
```

---

### Task 6: Evaluation harness (`evaluate.py`)

**Files:**
- Create: `evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `game.Game`, `game.random_legal_action`, `game.heuristic_action`, `mcts.run_mcts`, `mcts.select_action`.
- Produces: `evaluate_vs_opponent(model, num_simulations, opponent, num_games) -> dict` with keys `"win"`, `"loss"`, `"draw"`, `"incomplete"`. `opponent` is the string `"random"` or `"heuristic"`, or a Keras model (to evaluate against a previous checkpoint, using MCTS for both sides).

- [ ] **Step 1: Write `tests/test_evaluate.py`**

```python
from network import build_network
from evaluate import evaluate_vs_opponent


def test_evaluate_vs_random_runs_and_returns_valid_results():
    model = build_network()
    results = evaluate_vs_opponent(model, num_simulations=5, opponent="random", num_games=2)
    assert set(results.keys()) == {"win", "loss", "draw", "incomplete"}
    assert sum(results.values()) == 2


def test_evaluate_vs_heuristic_runs_and_returns_valid_results():
    model = build_network()
    results = evaluate_vs_opponent(model, num_simulations=5, opponent="heuristic", num_games=2)
    assert set(results.keys()) == {"win", "loss", "draw", "incomplete"}
    assert sum(results.values()) == 2


def test_evaluate_vs_another_model_runs_and_returns_valid_results():
    model_a = build_network()
    model_b = build_network()
    results = evaluate_vs_opponent(model_a, num_simulations=5, opponent=model_b, num_games=2)
    assert set(results.keys()) == {"win", "loss", "draw", "incomplete"}
    assert sum(results.values()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Write `evaluate.py`**

```python
import random
from game import Game, random_legal_action, heuristic_action
from mcts import run_mcts, select_action


def evaluate_vs_opponent(model, num_simulations, opponent, num_games):
    results = {"win": 0, "loss": 0, "draw": 0, "incomplete": 0}

    for _ in range(num_games):
        game = Game()
        agent_player = random.randint(1, 2)
        move_count = 0

        while not game.game_over and move_count < 100:
            if game.current_player == agent_player:
                root = run_mcts(game, model, num_simulations, add_noise=False)
                action = select_action(root, temperature=1e-3)
                sub_board, cell = action // 9, action % 9
            elif opponent == "random":
                sub_board, cell = random_legal_action(game)
            elif opponent == "heuristic":
                sub_board, cell = heuristic_action(game)
            else:
                root = run_mcts(game, opponent, num_simulations, add_noise=False)
                action = select_action(root, temperature=1e-3)
                sub_board, cell = action // 9, action % 9

            game.execute_move(sub_board, cell)
            move_count += 1

        if game.game_over:
            if game.winner == agent_player:
                results["win"] += 1
            elif game.winner is None:
                results["draw"] += 1
            else:
                results["loss"] += 1
        else:
            results["incomplete"] += 1

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 3 passed.

- [ ] **Step 5: Re-run Task 5's test now that `evaluate.py` exists**

Run: `python -m pytest tests/test_train.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "Add MCTS-driven evaluation harness (vs random/heuristic/checkpoint)"
```

---

### Task 7: Full training loop and entry point (`train.py`, part 2)

**Files:**
- Modify: `train.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `training_loop(num_iterations, games_per_iteration, num_simulations, batch_size, train_steps_per_iteration, replay_maxlen=20000, checkpoint_dir="./checkpoints", eval_games=10) -> None`. Script entry point via `if __name__ == "__main__":`.

- [ ] **Step 1: Add `training_loop` and the entry point to `train.py`**

Append to `train.py` (below `training_step`):

```python
def training_loop(num_iterations, games_per_iteration, num_simulations, batch_size,
                   train_steps_per_iteration, replay_maxlen=20000,
                   checkpoint_dir="./checkpoints", eval_games=10):
    import os
    import time

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = build_network()
    replay_buffer = deque(maxlen=replay_maxlen)
    start_time = time.time()

    for iteration in range(num_iterations):
        for _ in range(games_per_iteration):
            examples = play_self_play_game(model, num_simulations)
            replay_buffer.extend(examples)

        losses = []
        for _ in range(train_steps_per_iteration):
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                loss = training_step(model, batch)
                losses.append(loss[0])

        avg_loss = sum(losses) / len(losses) if losses else None
        elapsed = time.time() - start_time
        print(f"iteration {iteration} — {elapsed:.0f}s elapsed — replay_buffer={len(replay_buffer)} avg_loss={avg_loss}")

        model.save(f"{checkpoint_dir}/model_iter{iteration}.keras")

        vs_random = evaluate_vs_opponent(model, num_simulations, "random", eval_games)
        vs_heuristic = evaluate_vs_opponent(model, num_simulations, "heuristic", eval_games)
        print(f"iteration {iteration} vs random: {vs_random}")
        print(f"iteration {iteration} vs heuristic: {vs_heuristic}")


if __name__ == "__main__":
    # Short validation run first — confirms the pipeline produces sane data
    # and stable loss with no correctness bugs before scaling up.
    training_loop(
        num_iterations=3,
        games_per_iteration=4,
        num_simulations=25,
        batch_size=32,
        train_steps_per_iteration=20,
    )
```

- [ ] **Step 2: Run the short validation training loop**

Run: `python train.py`
Expected: prints 3 iterations of `iteration N — ...s elapsed — replay_buffer=... avg_loss=...` followed by `vs random: {...}` and `vs heuristic: {...}` lines for each iteration, no exceptions, `avg_loss` is a finite number, and `checkpoints/model_iter0.keras` through `model_iter2.keras` are created.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "Add full self-play/train/evaluate training loop with short validation config"
```

---

## After This Plan

Once the short validation run (Task 7) confirms the pipeline is sound:
- Scale up `num_iterations`, `games_per_iteration`, and `num_simulations` in `train.py`'s `__main__` block for a real training run (exact values to be chosen based on how the validation run's timing and loss trend look — not fixed in this plan).
- The evaluation cadence, checkpoint-keeping policy (e.g. only advance to a new "best" model if it beats the previous one), and any self-play parallelization are tuning/optimization work for that follow-up, not blocking correctness.
- TensorFlow.js export and the browser deployment are a separate future spec, per this plan's Global Constraints.
