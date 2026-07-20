# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An Ultimate Tic-Tac-Toe AI, trained via self-play with Monte Carlo Tree Search (MCTS)
guided by a policy/value network — an AlphaZero-style approach. Built for the Stardance
challenge; the trained model is meant to eventually be exported to TensorFlow.js and
deployed as a browser opponent (not yet implemented — see "Not yet built" below).

An earlier DQN-based approach was abandoned after extensive tuning (see git history
around commit `08a63a2` and `devlog.md`) — it kept converging to a near-input-invariant
policy instead of learning real tactics. MCTS was adopted instead because it uses actual
tree search for tactical lookahead (forced wins/blocks are found by search itself, not
inferred by the network), which is a better fit for this class of perfect-information
board game and sidesteps the failure mode DQN hit.

## Commands

Activate the venv first (Windows): `.venv\Scripts\activate`

- Install deps: `pip install -r requirements.txt`
- Run all tests: `python -m pytest tests/ -v`
- Run one test file: `python -m pytest tests/test_mcts.py -v`
- Run one test: `python -m pytest tests/test_mcts.py::test_mcts_finds_forced_win -v`
- Run tests matching a name: `python -m pytest tests/ -v -k leaf_parallel`
- Train (real run, resumes automatically if `final_model.keras` exists): `python train.py`

Tests import top-level modules (`from game import Game`, etc.), so always run pytest
from the repo root — that's what puts it on `sys.path`.

## Architecture

Six focused modules, each with a single responsibility, wired together in `train.py`:

- **`game.py`** — `Game`: pure game engine/rules for Ultimate Tic-Tac-Toe (9 sub-boards
  of 9 cells; win a sub-board, win 3 sub-boards in a line to win the meta-board; playing
  in cell N sends the opponent to sub-board N, or anywhere if that board is full/won).
  `Game.clone()` is a fast manual copy (not `deepcopy`) used heavily by MCTS to simulate
  without mutating the real game state. Also has `legal_action_mask`, `random_legal_action`,
  `heuristic_action` (a 1-ply lookahead: take an immediate win, else block one, else
  random — used as a training/evaluation opponent, not part of the trained agent).
- **`network.py`** — `build_network()`: the policy/value model. Three inputs (board
  occupancy planes, sub-board status, legal sub-board mask), two outputs: a scalar value
  in `[-1, 1]` (tanh) and an 81-way move-probability policy (softmax). `encode_state(game,
  perspective_player)` converts a `Game` into these three input arrays, always from a
  given player's perspective (defaults to whoever's turn it is).
- **`mcts.py`** — PUCT-style search. `MCTSNode` holds `.visit_count`/`.value_sum`/
  `.children`. Three search entry points, all producing an equivalent root `MCTSNode`:
  - `run_mcts` — one game, one network call per simulation.
  - `run_mcts_batch` — many simultaneous games, batching each round's leaf evaluations
    into one network call across all active games. Used by training/self-play, where many
    games are naturally in flight.
  - `run_mcts_leaf_parallel` — one game, batches multiple leaf evaluations per round via
    **virtual loss** (a temporary penalty applied along each walk's path so consecutive
    walks in the same round spread across different leaves instead of collapsing onto
    one). This is the only one of the three that helps a *single* live game, which is the
    deployment scenario — nothing to batch across when there's only one game in flight.
  - **Critical, non-obvious convention**: `Game.execute_move` does *not* flip
    `current_player` when a move ends the game (the winner stays `current_player`).
    Terminal MCTS values must therefore be computed from the perspective of whoever's
    turn it *would* conceptually be next (the loser: `-1.0` if there's a winner, `0.0`
    for a draw), not the winner's own perspective — otherwise the alternating-sign
    backup convention (`q = -child.value()` in `select_child`, negating `value` each
    level in the backup loop) breaks silently on exactly the moves that matter most. Get
    this sign wrong and MCTS stops finding forced wins/blocks; `test_mcts_finds_forced_win`
    /`test_mcts_finds_forced_block_in_narrow_position` exist specifically to catch this
    class of bug (they must pass even with a freshly-initialized, untrained network,
    since real search — not network judgment — is what finds a forced tactic).
- **`self_play.py`** — `play_self_play_batch(model, num_games, num_simulations, ...)`
  runs many games in lockstep via `run_mcts_batch`, returns training examples
  `(board, status, legal, policy_target, value_target)`. `policy_target` is MCTS's
  visit-count distribution over the 81 actions; `value_target` is the actual game
  outcome (`+1`/`-1`/`0`) from that position's mover's perspective, back-filled once the
  game ends.
- **`evaluate.py`** — `evaluate_vs_opponent(model, num_simulations, opponent, num_games)`
  plays the agent (MCTS-driven) against `"random"`, `"heuristic"`, or another model
  (also MCTS-driven), batched across `num_games` simultaneous games the same way
  self-play is. Returns `{"win", "loss", "draw", "incomplete"}` counts.
- **`train.py`** — `training_step` (one supervised gradient step: policy
  cross-entropy + value MSE) and `training_loop` (the outer loop: self-play → train →
  checkpoint → evaluate, repeated). Not standard supervised training against a fixed
  dataset — this is the standard AlphaZero training loop, where the network's own
  current weights generate the next batch of training data via MCTS self-play.

### Checkpointing and resuming

`training_loop` persists three things every iteration, all needed to resume correctly:
`checkpoints/model_iter{N}.keras` (numbered, permanent), `final_model.keras` (always the
latest, auto-updated — this is what `python train.py`'s `__main__` looks for to decide
whether to resume), and `checkpoints/replay_buffer.pkl` (the accumulated self-play data;
without this a resume would restart from an empty, low-diversity buffer even though the
network weights carried over). `_next_start_iteration` derives the correct next
iteration number from the highest `model_iter{N}.keras` already on disk, so re-invoking
`python train.py` repeatedly (e.g. looping it overnight) chains correctly instead of
colliding at a fixed offset.

### Testing conventions

Every test file mirrors its module 1:1 (`mcts.py` ↔ `tests/test_mcts.py`, etc.).
Two scenario builders in `tests/test_mcts.py` — `_make_immediate_win_scenario` and
`_make_forced_block_scenario` — construct hand-crafted board positions with a genuine
forced tactic (completing an actual *meta*-line, not just a sub-board — an easy mistake,
since winning a sub-board alone doesn't end the game) and are reused across all three
MCTS entry points' correctness tests. When benchmarking search speed (not correctness),
avoid these forced-win positions: once a plain PUCT search finds the win, it can revisit
that terminal node for free (no network call needed), which makes trivial positions a
misleading choice for timing comparisons — use a fresh `Game()` instead.

Tests that reload a freshly-built or freshly-loaded model and time it against another
call should warm both up first (one throwaway call each) before timing — the first call
at a given batch size pays TensorFlow's graph-tracing cost, which can dominate a small
benchmark and make whichever function runs second look artificially slower.

## Not yet built

TensorFlow.js export, the JavaScript MCTS port, and the browser UI are all future work —
no code for any of it exists yet. `run_mcts_leaf_parallel`'s virtual-loss batching exists
specifically to validate that algorithm in Python ahead of porting it to JS, since it's
the only batching strategy that helps a single live game (as opposed to the many
simultaneous games available during training).
