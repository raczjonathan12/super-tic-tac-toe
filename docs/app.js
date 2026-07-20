(async function () {
  const { Game } = window.GameModule;
  const { loadModel, predictBatch } = window.NetworkModule;
  const { runMctsLeafParallel, selectAction } = window.MctsModule;

  const NUM_SIMULATIONS = 96; // multiple of LEAF_BATCH_SIZE: every MCTS round
  const LEAF_BATCH_SIZE = 8;  // uses the same batch shape, so TF.js only ever
                              // compiles its kernels for that one shape (plus
                              // batch size 1 for the root expansion), instead
                              // of paying a one-time compile cost mid-game for
                              // a leftover partial-batch shape.
  const MIN_THINKING_MS = 700; // floor on how long the "AI is thinking" state
                                // is shown, so a fast search (~200ms) doesn't
                                // make the AI's move feel like it snaps in
                                // before the user can register it happened.

  UI.initRulesModal();

  let model;
  try {
    model = await loadModel('model/model.json');
  } catch (err) {
    UI.setStatus('Failed to load model: ' + err.message);
    return;
  }

  // Warm up every batch shape MCTS can hand to predictBatch. Root expansion
  // always uses batch size 1, but leaf-evaluation rounds can be smaller than
  // LEAF_BATCH_SIZE too: whenever several simulated walks in a round hit a
  // terminal position (common near the endgame), they need no network call,
  // so the actual batch of leaves left to evaluate can be any size from 1 up
  // to LEAF_BATCH_SIZE. Warming up all of them here, while the loading
  // overlay is still showing, moves every one-time TF.js kernel-compilation
  // cost out of the middle of a real game.
  const warmupGame = new Game();
  for (let size = 1; size <= LEAF_BATCH_SIZE; size++) {
    await predictBatch(model, new Array(size).fill(warmupGame));
  }

  UI.hideLoadingOverlay();

  let game, humanPlayer, lastMove, lastAiMove, gameOverHandled;

  const SCORE_KEY = 'super-ttt-score';
  let score;
  try {
    score = JSON.parse(localStorage.getItem(SCORE_KEY)) || { human: 0, ai: 0, draws: 0 };
  } catch (err) {
    score = { human: 0, ai: 0, draws: 0 };
  }
  UI.setScore(score);

  function recordResult() {
    if (game.winner === humanPlayer) score.human++;
    else if (game.winner) score.ai++;
    else score.draws++;
    UI.setScore(score);
    try {
      localStorage.setItem(SCORE_KEY, JSON.stringify(score));
    } catch (err) {
      // localStorage unavailable (e.g. private browsing); score just won't persist.
    }
  }

  function startNewGame() {
    game = new Game();
    humanPlayer = Math.random() < 0.5 ? 1 : 2;
    // humanPlayer is independent of game.currentPlayer (which side moves
    // first is decided separately by Game's own constructor); if the AI
    // happens to move first, refresh() below sends it straight into
    // runAiTurn().
    lastMove = null;
    lastAiMove = null;
    gameOverHandled = false;
    refresh();
  }

  function refresh() {
    UI.renderBoard(game, onHumanCellClick, lastMove, lastAiMove, game.currentPlayer === humanPlayer);
    if (game.gameOver) {
      if (!gameOverHandled) {
        gameOverHandled = true;
        recordResult();
      }
      if (game.winner === humanPlayer) UI.setStatus('You win!');
      else if (game.winner) UI.setStatus('AI wins!');
      else UI.setStatus("It's a draw!");
      UI.setFigureState('human', 'idle');
      UI.setFigureState('robot', 'idle');
      return;
    }
    if (game.currentPlayer === humanPlayer) {
      UI.setStatus('Your turn');
      UI.setFigureState('human', 'active');
      UI.setFigureState('robot', 'idle');
    } else {
      UI.setStatus('AI is thinking...');
      UI.setFigureState('robot', 'thinking');
      UI.setFigureState('human', 'idle');
      runAiTurn();
    }
  }

  function onHumanCellClick(subBoard, cell) {
    if (game.gameOver || game.currentPlayer !== humanPlayer) return;
    game.executeMove(subBoard, cell);
    lastMove = { subBoard, cell };
    refresh();
  }

  async function runAiTurn() {
    const evaluateFn = (games) => predictBatch(model, games);
    const searchStart = Date.now();
    const root = await runMctsLeafParallel(game, evaluateFn, NUM_SIMULATIONS, LEAF_BATCH_SIZE, 1.5);
    const elapsed = Date.now() - searchStart;
    if (elapsed < MIN_THINKING_MS) {
      await new Promise((resolve) => setTimeout(resolve, MIN_THINKING_MS - elapsed));
    }
    const action = selectAction(root, 1e-3);
    const subBoard = Math.floor(action / 9), cell = action % 9;
    game.executeMove(subBoard, cell);
    lastMove = { subBoard, cell };
    lastAiMove = { subBoard, cell };
    refresh();
  }

  document.getElementById('new-game-button').addEventListener('click', startNewGame);

  startNewGame();
})();
