(function (root) {

function renderBoard(game, onCellClick, lastMove, lastAiMove, isHumanTurn) {
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';

  const legalSet = new Set(game.legalCoords.map(([sb, c]) => `${sb}-${c}`));
  const legalSubBoards = new Set(game.legalCoords.map(([sb]) => sb));
  // Only show legal-move highlighting and accept clicks while it's actually
  // the human's turn -- otherwise cells stay lit up and clickable-looking
  // during the AI's turn even though a click would silently no-op, which
  // reads as broken rather than merely "not your turn".
  const interactive = isHumanTurn && !game.gameOver;

  for (let sb = 0; sb < 9; sb++) {
    const subEl = document.createElement('div');
    subEl.className = 'subboard';
    if (interactive && legalSubBoards.has(sb)) subEl.classList.add('legal');
    if (game.subBoardsStatus[sb][1] === 1) subEl.classList.add('won-1');
    if (game.subBoardsStatus[sb][2] === 1) subEl.classList.add('won-2');

    for (let c = 0; c < 9; c++) {
      const cellEl = document.createElement('div');
      cellEl.className = 'cell';
      const [open, p1, p2] = game.board[sb][c];
      if (p1) { cellEl.textContent = 'X'; cellEl.classList.add('p1'); }
      else if (p2) { cellEl.textContent = 'O'; cellEl.classList.add('p2'); }

      if (lastMove && lastMove.subBoard === sb && lastMove.cell === c) {
        cellEl.classList.add('pop-in');
      }
      if (lastAiMove && lastAiMove.subBoard === sb && lastAiMove.cell === c) {
        cellEl.classList.add('ai-last-move');
      }

      const isLegal = interactive && legalSet.has(`${sb}-${c}`);
      if (!isLegal) {
        cellEl.classList.add('disabled');
      } else {
        cellEl.addEventListener('click', () => onCellClick(sb, c));
      }
      subEl.appendChild(cellEl);
    }
    boardEl.appendChild(subEl);
  }
}

function setStatus(text) {
  document.getElementById('status-banner').textContent = text;
}

function setFigureState(figure, state) {
  const id = figure === 'human' ? 'human-figure' : 'robot-figure';
  const el = document.getElementById(id);
  el.classList.remove('idle', 'thinking', 'active');
  el.classList.add(state);
}

function hideLoadingOverlay() {
  document.getElementById('loading-overlay').classList.add('hidden');
}

function setScore(score) {
  document.getElementById('score-human').textContent = score.human;
  document.getElementById('score-ai').textContent = score.ai;
  document.getElementById('score-draws').textContent = score.draws;
}

function initRulesModal() {
  const modal = document.getElementById('rules-modal');
  document.getElementById('rules-button').addEventListener('click', () => {
    modal.classList.remove('hidden');
  });
  document.getElementById('rules-modal-close').addEventListener('click', () => {
    modal.classList.add('hidden');
  });
  modal.classList.remove('hidden'); // show on first load
}

root.UI = { renderBoard, setStatus, setFigureState, hideLoadingOverlay, initRulesModal, setScore };

})(typeof window !== 'undefined' ? window : this);
