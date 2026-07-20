# Virtual-Loss Leaf-Parallel MCTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `run_mcts_leaf_parallel` to `mcts.py` — a single-game MCTS variant that batches multiple leaf evaluations into one network call per round via virtual loss, instead of one call per simulation.

**Architecture:** Within each round, walk the tree `leaf_batch_size` times before touching the network. Each walk applies a temporary virtual loss to every node it selects through (discouraging the next walk in the same round from picking the identical path), collecting up to `leaf_batch_size` distinct leaves. Those leaves are evaluated in one batched network call, then every walk's virtual loss is undone and replaced with its real backed-up value.

**Tech Stack:** Python, NumPy. Reuses existing `MCTSNode`, `select_child`, `batch_evaluate_and_expand`, `_add_root_noise`, `_terminal_or_none` from `mcts.py` unchanged.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-virtual-loss-mcts-design.md`
- `run_mcts` and `run_mcts_batch` are not modified — zero risk to their existing, already-tested behavior.
- Not wired into `self_play.py`, `evaluate.py`, or `train.py` — standalone addition to `mcts.py` only.
- No changes to `game.py` or `network.py`.
- Run tests from the repo root with `python -m pytest tests/test_mcts.py -v`.
- Use the project's existing venv (`.venv`) for all commands.

---

### Task 1: `run_mcts_leaf_parallel`

**Files:**
- Modify: `mcts.py`
- Modify: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `MCTSNode`, `select_child(node, c_puct)`, `batch_evaluate_and_expand(nodes, games, model)`, `evaluate_and_expand(node, game, model)`, `_add_root_noise(root, dirichlet_alpha, dirichlet_epsilon)`, `_terminal_or_none(sim_game)` — all already defined in `mcts.py`.
- Produces: `run_mcts_leaf_parallel(game, model, num_simulations, leaf_batch_size=8, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_epsilon=0.25, add_noise=False) -> MCTSNode` (the root, same return type/shape as `run_mcts`/each element of `run_mcts_batch`'s return list — has `.children`, each child has `.visit_count`, usable directly with the existing `get_policy_target(root)` and `select_action(root, temperature)`).

- [ ] **Step 1: Write the failing correctness tests**

Append to `tests/test_mcts.py` (the file already has `_make_immediate_win_scenario` and `_make_forced_block_scenario` from Task 3 of the prior plan — reuse them, do not redefine):

```python
from mcts import run_mcts_leaf_parallel


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
    from mcts import run_mcts

    model = build_network()
    game, _ = _make_immediate_win_scenario()
    num_simulations = 200

    start = time.time()
    run_mcts(game, model, num_simulations=num_simulations, add_noise=False)
    sequential_time = time.time() - start

    start = time.time()
    run_mcts_leaf_parallel(game, model, num_simulations=num_simulations, leaf_batch_size=8, add_noise=False)
    parallel_time = time.time() - start

    assert parallel_time < sequential_time
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mcts.py -v -k leaf_parallel`
Expected: FAIL with `ImportError: cannot import name 'run_mcts_leaf_parallel' from 'mcts'`

- [ ] **Step 3: Implement `run_mcts_leaf_parallel` in `mcts.py`**

Add this function after `run_mcts_batch` (the existing last function in the file, `select_action`, currently comes before `run_mcts_batch`'s end — append this new function after `run_mcts_batch`'s closing `return roots`):

```python
def run_mcts_leaf_parallel(game, model, num_simulations, leaf_batch_size=8, c_puct=1.5,
                            dirichlet_alpha=0.3, dirichlet_epsilon=0.25, add_noise=False):
    """Single-game MCTS that batches leaf evaluations within one tree via
    virtual loss, instead of run_mcts_batch's batching across many
    simultaneous games. Each round walks the tree up to leaf_batch_size
    times before making one network call, applying a temporary virtual
    loss along each walk's path so consecutive walks in the same round
    naturally spread across different leaves. Semantics (PUCT selection,
    terminal-value convention, real backup) are identical to run_mcts;
    this only changes how leaf evaluations are scheduled and dispatched."""
    root = MCTSNode(prior=1.0)
    evaluate_and_expand(root, game, model)

    if add_noise:
        _add_root_noise(root, dirichlet_alpha, dirichlet_epsilon)

    simulations_done = 0
    while simulations_done < num_simulations:
        batch_size = min(leaf_batch_size, num_simulations - simulations_done)

        paths = []
        pending_values = [None] * batch_size
        leaf_positions = []
        leaf_nodes = []
        leaf_games = []

        for i in range(batch_size):
            node = root
            sim_game = game.clone()
            path = [node]

            while node.is_expanded() and not sim_game.game_over:
                action, node = select_child(node, c_puct)
                # Virtual loss: make this child look artificially good to
                # its own mover, so select_child's q = -child.value() at
                # the parent discourages picking it again this round.
                node.visit_count += 1
                node.value_sum += 1.0
                sub_board, cell = action // 9, action % 9
                sim_game.execute_move(sub_board, cell)
                path.append(node)

            paths.append(path)
            terminal_value = _terminal_or_none(sim_game)
            if terminal_value is not None:
                pending_values[i] = terminal_value
            else:
                leaf_positions.append(i)
                leaf_nodes.append(node)
                leaf_games.append(sim_game)

        if leaf_nodes:
            values = batch_evaluate_and_expand(leaf_nodes, leaf_games, model)
            for pos, v in zip(leaf_positions, values):
                pending_values[pos] = v

        for path, value in zip(paths, pending_values):
            # Undo this walk's virtual loss (root, at path[0], never got
            # any -- only nodes actually selected as a child do), then
            # apply the real backup exactly as run_mcts does.
            for node in path[1:]:
                node.visit_count -= 1
                node.value_sum -= 1.0
            for path_node in reversed(path):
                path_node.value_sum += value
                path_node.visit_count += 1
                value = -value

        simulations_done += batch_size

    return root
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_mcts.py -v -k leaf_parallel`
Expected: 5 passed.

If `test_run_mcts_leaf_parallel_finds_forced_win` or `_finds_forced_block` fail: the bug is almost certainly a sign error in the virtual loss update (`node.value_sum += 1.0` should make the search *less* likely to reselect that child, not more — if the search is instead ignoring the win/block, try flipping the sign to `-= 1.0` and re-derive from `select_child`'s `q = -child.value()` which direction actually discourages reselection). If `test_run_mcts_leaf_parallel_total_visits_match_num_simulations` fails: check that the undo loop (`path[1:]`) and the real backup loop (`reversed(path)`) are both operating on the same `path` list and that `simulations_done` increments by `batch_size` (not a fixed `leaf_batch_size`) so the final partial round is counted correctly.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (37 total: the 32 existing plus these 5 new ones).

- [ ] **Step 6: Commit**

```bash
git add mcts.py tests/test_mcts.py
git commit -m "$(cat <<'EOF'
Add virtual-loss leaf-parallel MCTS for single-game batching

run_mcts_batch batches network calls across many simultaneous games,
which is what self-play/evaluation use during training, but doesn't
help the eventual single-live-game deployment case (one browser game
against one human -- nothing to batch across).

run_mcts_leaf_parallel batches within a single game's tree instead: each
round collects up to leaf_batch_size leaves via virtual loss (a
temporary penalty applied along each walk's path so consecutive walks
in the same round spread across different leaves) before making one
batched network call, cutting network round-trips roughly
leaf_batch_size-fold for a single game. Validated against the same
forced-win/forced-block scenarios used for run_mcts and run_mcts_batch,
plus a call-count test and a direct timing comparison against run_mcts.

Not wired into self_play.py/evaluate.py/train.py -- those already batch
across simultaneous games during training, which beats this. This
function exists to validate the algorithm ahead of porting it to the
future TensorFlow.js deployment, where it's the only batching option.
EOF
)"
```

---

## After This Plan

- Porting this algorithm to JavaScript/TensorFlow.js for the actual browser deployment is separate future work, per the design spec's "Out of scope" section.
- Tuning `leaf_batch_size` for the eventual deployment's real latency budget (the design's default of 8 was chosen as a reasonable starting point, not measured against a browser target) is also future work, once TF.js timing is actually measurable.
