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

  let game = new Game();
  const humanPlayer = Math.random() < 0.5 ? 1 : 2;
  // humanPlayer is independent of game.currentPlayer (which side moves
  // first is decided separately by Game's own constructor); if the AI
  // happens to move first, refresh() below sends it straight into
  // runAiTurn().

  function refresh() {
    UI.renderBoard(game, onHumanCellClick);
    if (game.gameOver) {
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
      setTimeout(runAiTurn, 30);
    }
  }

  function onHumanCellClick(subBoard, cell) {
    if (game.gameOver || game.currentPlayer !== humanPlayer) return;
    game.executeMove(subBoard, cell);
    refresh();
  }

  async function runAiTurn() {
    const evaluateFn = (games) => predictBatch(model, games);
    const root = await runMctsLeafParallel(game, evaluateFn, NUM_SIMULATIONS, LEAF_BATCH_SIZE, 1.5);
    const action = selectAction(root, 1e-3);
    game.executeMove(Math.floor(action / 9), action % 9);
    refresh();
  }

  refresh();
})();
