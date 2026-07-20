(async function () {
  const { Game } = window.GameModule;
  const { loadModel, predictBatch } = window.NetworkModule;
  const { runMctsLeafParallel, selectAction } = window.MctsModule;

  const NUM_SIMULATIONS = 100;
  const LEAF_BATCH_SIZE = 8;

  UI.initRulesModal();

  let model;
  try {
    model = await loadModel('model/model.json');
  } catch (err) {
    UI.setStatus('Failed to load model: ' + err.message);
    return;
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
