# Virtual-Loss Leaf-Parallel MCTS

## Context

The trained agent (see `docs/superpowers/specs/2026-07-19-alphazero-mcts-design.md` and
`docs/superpowers/plans/2026-07-19-alphazero-mcts.md`) uses `mcts.run_mcts_batch` to
batch network calls across many *simultaneous games* during self-play and evaluation.
That works well during training, where dozens of games run in parallel, but it does
nothing for the eventual deployment scenario: a single live game against one human,
where there is only ever one game in flight. In that case, `run_mcts`'s current design
makes one individual network call per simulation — for `num_simulations=400` (the depth
we measured as meaningfully stronger against the heuristic baseline, see prior session),
that's 400 separate round-trips per move, which is far too slow for a browser deployment
targeting sub-second move times.

## Goal

Add a single-game MCTS variant that batches multiple leaf evaluations into one network
call per round, cutting network round-trips roughly `leaf_batch_size`-fold for a single
game, without changing what the search concludes. This validates the algorithm in Python
now, ahead of porting the same logic to a JavaScript/TensorFlow.js implementation later
(deployment itself is out of scope for this spec).

## Approach: virtual loss

Standard technique for parallelizing MCTS simulations within a single tree. Instead of
walking the tree once and evaluating one leaf per simulation, walk it `leaf_batch_size`
times in a row *before* touching the network:

1. Each walk-through applies a temporary **virtual loss** to every node it passes
   through — a fake, immediate negative value applied to that node's stats — which
   discourages PUCT from selecting the exact same path again on the very next walk
   within the same round. This is what causes consecutive walks to naturally spread
   across *different* leaves instead of collapsing onto the same one.
2. Once `leaf_batch_size` leaves have been collected this way (fewer if the tree is too
   small to produce that many distinct leaves), evaluate all of them in a single batched
   network call.
3. Back up every result: remove that leaf's virtual loss and replace it with the real
   backed-up value, same alternating-sign convention as `run_mcts`/`run_mcts_batch`.

This changes *how fast* the search reaches a given number of simulations, not what
those simulations conclude — virtual loss is a scheduling trick, not a change to the
value/backup semantics already validated in `run_mcts`.

## Scope

- **New function** `run_mcts_leaf_parallel(game, model, num_simulations, leaf_batch_size=8, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, add_noise=False) -> MCTSNode`
  in `mcts.py`, alongside (not replacing) the existing `run_mcts`. Reuses `MCTSNode`,
  `select_child`, `batch_evaluate_and_expand`, `_add_root_noise`, `_terminal_or_none` —
  only the per-round structure changes.
- `run_mcts` itself is untouched: zero risk of regressing `play_self_play_game` or
  anything else that already depends on its exact current behavior.
- **Not wired into `self_play.py`, `evaluate.py`, or `train.py`.** Those already batch
  effectively across many simultaneous games during training/evaluation, which beats
  single-tree virtual-loss batching when many games are available. This function exists
  for the future single-live-game deployment path, validated here in Python first.
- No changes to `game.py` or `network.py`.

## Testing / validation

Same TDD pattern as the rest of this codebase, reusing the existing tactical scenarios
from `tests/test_mcts.py` (`_make_immediate_win_scenario`, `_make_forced_block_scenario`)
so correctness is checked against known-correct answers, not just "does it run":

- **Correctness**: `run_mcts_leaf_parallel` must still find the forced win and the
  forced block in those scenarios, same as `run_mcts` and `run_mcts_batch` already do.
  Virtual loss must not change the conclusion.
- **Speedup is real**: a call-counting test confirms the number of network calls is
  roughly `num_simulations / leaf_batch_size` (not `num_simulations`), and a direct
  timing comparison against `run_mcts` on the same scenario shows a measurable
  improvement.

## Out of scope

- Wiring `run_mcts_leaf_parallel` into `self_play.py`, `evaluate.py`, or `train.py`.
- Any TensorFlow.js / JavaScript work. This spec is Python-only, preparing an algorithm
  that a future JS port will mirror.
- Changes to `game.py`, `network.py`, or the existing `run_mcts`/`run_mcts_batch`.
