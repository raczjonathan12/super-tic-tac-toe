# Browser Frontend for the Stardance Challenge

## Context

The project trains an AlphaZero-style Ultimate Tic-Tac-Toe agent (`game.py`, `network.py`,
`mcts.py`) via self-play (see `docs/superpowers/specs/2026-07-19-alphazero-mcts-design.md`
and `docs/superpowers/specs/2026-07-20-virtual-loss-mcts-design.md`). CLAUDE.md's "Not yet
built" section names three remaining pieces: TensorFlow.js export, a JavaScript MCTS port,
and a browser UI. This spec covers all three, to be deployed as a static site on GitHub
Pages for the Stardance challenge.

`run_mcts_leaf_parallel` (in `mcts.py`) already validates, in Python, the virtual-loss
batching algorithm this needs — a single live game has no other games to batch across, so
leaf-parallel batching within one tree is the only strategy that helps. This spec ports
that same algorithm to JavaScript.

## Goal

A plain HTML/CSS/JS page, deployable on GitHub Pages, where a human plays Ultimate
Tic-Tac-Toe against the trained model running entirely client-side via TensorFlow.js.
Each AI move must complete in well under 1.5s. The UI should be playful and visually
polished: a dark neon theme, an animated human-vs-robot hero with glowing eyes that react
to game state, a rules popup, and clear legal-move highlighting on the board.

## Scope

### Model conversion
- Install the `tensorflowjs` pip package in the project venv.
- Convert `final_model.keras` to the TF.js layers format via
  `tensorflowjs_converter --input_format=keras final_model.keras docs/model/`.
- Commit the resulting `model.json` + weight shard(s) to `docs/model/`. No server-side
  inference — the browser loads these files directly and runs the model with `tfjs`.

### File layout
All new frontend code lives under `docs/`, so GitHub Pages can serve it directly from
that folder on `master` with no build step, and the existing Python training code at the
repo root is untouched.

```
docs/
  index.html
  style.css
  app.js
  game.js
  network.js
  mcts.js
  ui.js
  model/          (model.json + weight shards, converted from final_model.keras)
  test/
    test_game.js
    test_mcts.js
```

### `game.js` — game engine port
Direct port of `game.py`'s `Game` class and module-level helpers: board state (9
sub-boards x 9 cells x {open, player1, player2}), `subBoardsStatus`, `subBoardsLegal`,
`legalMoves()`, `executeMove(subBoard, cell)`, `checkWin(...)`, `legalActionMask(game)`.
Pure logic, no DOM access, so it can be unit-tested standalone (see Testing below).
Mirrors the Python semantics exactly, including the non-obvious convention documented in
CLAUDE.md: `executeMove` does not flip `currentPlayer` when a move ends the game.

### `network.js` — model loading and encoding
- Loads the TF.js model once at page startup (`tf.loadLayersModel('model/model.json')`),
  shown behind a loading indicator.
- `encodeState(game, perspectivePlayer)` — direct port of `network.py`'s `encode_state`,
  producing the same three input arrays (board occupancy planes, sub-board status, legal
  sub-board mask) from a given player's perspective.
- `predictBatch(games)` — stacks encoded inputs for multiple games/leaves into batched
  tensors and calls `model.predict(...)` once, mirroring
  `mcts.py::batch_evaluate_and_expand`'s single-call-per-round batching.

### `mcts.js` — leaf-parallel MCTS port
Direct port of `run_mcts_leaf_parallel` from `mcts.py`: `MCTSNode` (prior, visitCount,
valueSum, children), `selectChild` (PUCT: `q = -child.value()` plus the exploration
term), virtual loss applied per-walk within a round to spread leaf selection, one batched
`network.js::predictBatch` call per round, then backup with the alternating-sign
convention. Same terminal-value convention as Python (`-1` if there's a winner, `0` for a
draw, evaluated from the perspective of whoever's turn it would conceptually be next).

Simulation budget: start at `numSimulations=100`, `leafBatchSize=8-16` (~10-15 batched
network calls per move). Benchmark in-browser against the ≤1.5s/move target once wired up
and tune down if needed — the network is small (2.6MB, small dense layers) so each batched
call should take low tens of ms.

### `ui.js` — board rendering and interaction
- Renders the 9x9 grid as nested sub-boards.
- Legal sub-boards (per `game.subBoardsLegal` and open status) get a glowing CSS outline;
  illegal ones are dimmed and non-interactive.
- Clicking a legal cell triggers the human's move, then hands off to `app.js` to run the
  AI's turn.
- Drives character animation state: `idle`, `thinking` (while MCTS runs for the AI's
  move), `human-turn` (glow on the human figure), `ai-turn` (glow on the robot figure).
- Rules modal: shown on first load, reopenable via a persistent "Rules" button, explains
  Ultimate Tic-Tac-Toe's rules (sub-board wins, meta-board win, forced sub-board routing),
  dismissable, styled to match the theme.

### `app.js` — orchestration
Turn loop: human is assigned a random side at page load (mirroring `Game.__init__`'s
`random.randint(1, 2)`); on the human's turn, waits for a UI click; on the AI's turn, sets
`thinking` animation state, runs `run_mcts_leaf_parallel`-equivalent search via `mcts.js`,
picks the highest-visit-count action (temperature 0, matching deployment-style greedy
play), executes it, updates the board and animation state.

### Visual design
Dark background, neon cyan/violet glow accents throughout (legal-cell outlines, character
eyes, active-turn indicators). Hero area above the board has a stylized human silhouette
and robot silhouette facing each other; both have CSS-animated glowing eyes (radial
gradient + keyframe pulse) that react to game state as described above — idle blink loop,
brighter pulse during AI "thinking", steady glow indicating whose turn it is. Built with
plain CSS keyframe animations, no animation libraries. Concrete palette/typography/spacing
choices will follow the `ui-ux-pro-max` skill's guidelines during implementation.

### Deployment
- GitHub Pages enabled via `gh` CLI (or repo settings), configured to serve from the
  `docs/` folder on `master`.
- No CI/build step: the TF.js model files are committed directly as static assets, so
  Pages serves the site as-is.

## Testing

Following this repo's TDD convention (`mcts.py` <-> `tests/test_mcts.py`), and reusing
the same tactical correctness scenarios already validated in Python
(`_make_immediate_win_scenario`, `_make_forced_block_scenario` from
`tests/test_mcts.py`):

- `docs/test/test_game.js` — plain Node script (ES modules, no framework/dependency)
  asserting `game.js`'s move execution, sub-board/meta-board win detection, and legal-move
  masking match expected outcomes.
- `docs/test/test_mcts.js` — asserts `mcts.js`'s `run_mcts_leaf_parallel` port finds the
  forced win and forced block in the ported scenarios, same as the Python tests require.
  Since these run in Node (not the browser) they'll need a lightweight stand-in for the
  network (e.g. a fixed/random policy+value function) rather than loading the real TF.js
  model, matching how `tests/test_mcts.py` uses a freshly-initialized untrained network
  for these specific correctness checks.
- Both run via plain `node docs/test/test_game.js` / `node docs/test/test_mcts.js` — no
  new project dependencies, kept separate from `docs/` files actually shipped to Pages
  (or excluded from what a Pages visitor would load, e.g. not linked from `index.html`).
- Manual verification: after wiring up the real model, play a full game in an actual
  browser, confirm move latency stays under the 1.5s target, confirm legal-move
  highlighting and the rules popup behave correctly.

## Out of scope

- Difficulty levels / adjustable simulation count exposed in the UI (fixed budget only).
- Human-vs-human local mode (human-vs-AI only, per prior decision).
- A GitHub Actions build/deploy workflow (model files are committed directly instead).
- Any changes to the existing Python training pipeline (`train.py`, `self_play.py`,
  `evaluate.py`) or the `run_mcts`/`run_mcts_batch`/`run_mcts_leaf_parallel` functions in
  `mcts.py` — this spec only adds a JS port alongside them, it doesn't modify them.
