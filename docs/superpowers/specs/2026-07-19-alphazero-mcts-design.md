# Ultimate Tic-Tac-Toe: AlphaZero-style MCTS + Policy/Value Network

## Context

The project previously used a DQN self-play approach (see git history at commit
`08a63a2` for the full attempt). Despite many targeted fixes — zero-sum Bellman
backup, Double DQN, potential-based reward shaping, a vectorized `BatchGame`
self-play/eval engine, hand-engineered tactical win/block input features, an
opponent pool + heuristic-in-training curriculum, and tiered priority/win replay
buffers — the network repeatedly converged to a near-input-invariant policy
(picking the same action across meaningfully different board states) instead of
learning real tactics. This is the classic sign that the algorithm, not the
tuning, was the problem: DQN asks a single forward pass to implicitly contain
all tactical lookahead, which is a poor fit for a game where "is there an
immediate win/block" reasoning matters constantly.

## Goal

Train a strong Ultimate Tic-Tac-Toe agent — a genuinely trained neural network,
via self-play — capable of beating a skilled human, that doesn't degenerate
into repetitive/non-discriminating play. The agent will eventually be deployed
client-side (TensorFlow.js, GitHub Pages, for the Stardance challenge), but
**that deployment step is explicit future work, not part of this spec.** This
spec covers only the Python self-play training pipeline.

## Approach

AlphaZero-style: Monte Carlo Tree Search (MCTS), guided by a small trained
policy/value network, generates self-play training data. The network is
trained via supervised learning on that data (not bootstrapped TD targets),
which avoids the self-referential collapse dynamic DQN kept hitting.

## Architecture

### Game engine (reused, unchanged)

`Game` (single-game, used for evaluation / reference) and `BatchGame`
(vectorized, used for batching leaf network evaluations across multiple
simultaneous MCTS trees) are reused as-is from the DQN version. Both are
already validated correct via 500-trial randomized equivalence testing
(see prior git history). No changes to game rules/logic in this spec.

### Network

Single Keras model, two output heads:

- **Value head**: scalar, `tanh` activation → range `[-1, 1]`. "How good is
  this position for the player to move," from that player's perspective.
- **Policy head**: 81 logits → softmax, masked to legal moves at
  consumption time (illegal-move mass renormalized away, not baked into the
  network's raw output). "Which moves look promising" — a *prior* that
  guides where MCTS spends simulation budget, not a final answer.

Inputs: board occupancy planes + sub-board status + legal mask (3 inputs,
matching the DQN version's original pre-tactical-features design). The
hand-engineered tactical win/block features from the DQN attempt are
**dropped** — MCTS's actual lookahead makes that signal redundant, and a
simpler/smaller network means faster inference once ported to TensorFlow.js.

### MCTS

PUCT variant (the AlphaZero flavor of UCT):

- Selection: descend the tree choosing the child maximizing
  `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`.
- Expansion: at a leaf, run the network once to get `(value, policy)`;
  create child nodes for all legal moves with prior probabilities from the
  policy head.
- Backup: propagate the leaf value up the path, negating at each ply
  (zero-sum alternation, same principle as the corrected DQN Bellman backup).
- Root exploration: add Dirichlet noise to the root's policy priors during
  self-play (not during evaluation/inference) so self-play games don't
  collapse onto a single deterministic line early in training.
- Simulation count: configurable; higher during training self-play is
  generally beneficial for data quality, lower at inference time to hit the
  <0.5s move-time target established for the eventual browser deployment.

### Self-play data generation

Per move, during self-play:

1. Run MCTS for `N` simulations from the current position.
2. The resulting child-visit-count distribution over legal moves becomes the
   policy training target for that position.
3. Sample the actual move played from that distribution using a temperature
   parameter — higher temperature (more random) early in the game, lower
   (more greedy) later, standard AlphaZero practice to balance exploration
   in the opening against decisive, learnable endgame play.
4. Store `(state, mcts_policy, mover)` for the position.

When the game ends, back-fill every stored position from that game with the
actual outcome from its own mover's perspective (`+1` win / `-1` loss / `0`
draw) as the value training target.

### Training loop

Iterative, standard AlphaZero cadence:

1. Generate a batch of self-play games using the *current* network + MCTS.
2. Train the network via supervised learning on that batch (plus a replay
   window of recent self-play data, not just the newest batch) —
   cross-entropy loss for the policy head against MCTS visit distributions,
   MSE/huber for the value head against game outcomes.
3. Evaluate the newly trained network (with MCTS) against: the previous
   checkpoint, the existing heuristic baseline, and random play — reusing
   the evaluation harness pattern from the DQN version (`evaluate`/
   `evaluate_batch`-style, adapted for MCTS-driven move selection instead of
   greedy Q-argmax).
4. Keep the new network if it's an improvement (or unconditionally advance
   on a fixed schedule — exact gating policy to be decided during
   implementation planning based on how noisy evaluation proves to be at
   small self-play-batch sizes).
5. Repeat.

Time budget: open-ended per user preference. Default to the pattern that
worked throughout the DQN effort — a short validation run first (a few dozen
self-play games) to confirm the pipeline produces sane data and stable loss
with no correctness bugs, *then* scale up to a longer real training run once
trusted.

## Testing / validation strategy

- **MCTS correctness**: unit-test against constructed positions with an
  obvious winning move — after sufficient simulations, that move should
  receive the large majority of visits. Directly analogous to the "obvious
  win" diagnostic that caught the DQN's collapse; same technique, now used
  as an upfront correctness check rather than a late diagnostic.
- **Ongoing training signal**: the vs-random / vs-heuristic / vs-previous-
  checkpoint evaluation benchmarks, carried over from the DQN version, track
  real progress across training iterations.
- **Game engine**: no changes, so no new equivalence testing needed there —
  prior 500-trial validation still holds.

## Out of scope

- TensorFlow.js export and the JavaScript MCTS/UI implementation for the
  actual browser deployment (Stardance challenge). Separate future spec.
- Any changes to `Game`/`BatchGame` game logic.
