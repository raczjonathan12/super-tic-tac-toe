(function (root) {

function renderBoard(game, onCellClick) {
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';

  const legalSet = new Set(game.legalCoords.map(([sb, c]) => `${sb}-${c}`));
  const legalSubBoards = new Set(game.legalCoords.map(([sb]) => sb));

  for (let sb = 0; sb < 9; sb++) {
    const subEl = document.createElement('div');
    subEl.className = 'subboard';
    if (legalSubBoards.has(sb) && !game.gameOver) subEl.classList.add('legal');
    if (game.subBoardsStatus[sb][1] === 1) subEl.classList.add('won-1');
    if (game.subBoardsStatus[sb][2] === 1) subEl.classList.add('won-2');

    for (let c = 0; c < 9; c++) {
      const cellEl = document.createElement('div');
      cellEl.className = 'cell';
      const [open, p1, p2] = game.board[sb][c];
      if (p1) { cellEl.textContent = 'X'; cellEl.classList.add('p1'); }
      else if (p2) { cellEl.textContent = 'O'; cellEl.classList.add('p2'); }

      const isLegal = legalSet.has(`${sb}-${c}`) && !game.gameOver;
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

root.UI = { renderBoard, setStatus, setFigureState, hideLoadingOverlay, initRulesModal };

})(typeof window !== 'undefined' ? window : this);
