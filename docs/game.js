(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.GameModule = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {

const WIN_LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];

function makeBoard() {
  const board = [];
  for (let sb = 0; sb < 9; sb++) {
    const row = [];
    for (let c = 0; c < 9; c++) row.push([1, 0, 0]);
    board.push(row);
  }
  return board;
}

function copyBoard(board) {
  return board.map(row => row.map(cell => cell.slice()));
}

class Game {
  constructor() {
    this.board = makeBoard();
    this.subBoardsStatus = Array.from({ length: 9 }, () => [1, 0, 0, 0]);
    this.subBoardsLegal = new Array(9).fill(1);
    this.currentPlayer = 1 + Math.floor(Math.random() * 2);
    this.gameOver = false;
    this.winner = null;
    this.legalMoves();
  }

  clone() {
    const g = Object.create(Game.prototype);
    g.board = copyBoard(this.board);
    g.subBoardsStatus = this.subBoardsStatus.map(s => s.slice());
    g.subBoardsLegal = this.subBoardsLegal.slice();
    g.currentPlayer = this.currentPlayer;
    g.gameOver = this.gameOver;
    g.winner = this.winner;
    g.legalCoords = this.legalCoords.map(pair => pair.slice());
    return g;
  }

  legalMoves() {
    const coords = [];
    for (let sb = 0; sb < 9; sb++) {
      const isOpen = this.subBoardsStatus[sb][0] === 1;
      const isLegalSb = this.subBoardsLegal[sb] === 1;
      if (!isOpen || !isLegalSb) continue;
      for (let c = 0; c < 9; c++) {
        if (this.board[sb][c][0] === 1) coords.push([sb, c]);
      }
    }
    this.legalCoords = coords;
  }

  checkWin(currentSubBoard, isMeta) {
    if (isMeta) {
      const metaBoard = this.subBoardsStatus.map(s => s[this.currentPlayer]);
      const hasMetaWin = WIN_LINES.some(line => line.every(i => metaBoard[i] === 1));
      if (hasMetaWin) {
        this.winner = this.currentPlayer;
        this.gameOver = true;
        return 'winner';
      }
      const anyOpen = this.subBoardsStatus.some(s => s[0] === 1);
      if (!anyOpen) {
        this.winner = null;
        this.gameOver = true;
        return 'draw';
      }
      return 'ongoing';
    }
    const sub = this.board[currentSubBoard].map(cell => cell[this.currentPlayer]);
    const hasSubWin = WIN_LINES.some(line => line.every(i => sub[i] === 1));
    if (hasSubWin) {
      this.subBoardsStatus[currentSubBoard][this.currentPlayer] = 1;
      this.subBoardsStatus[currentSubBoard][0] = 0;
      return 'cell_win';
    }
    const anyOpenCell = this.board[currentSubBoard].some(cell => cell[0] === 1);
    if (!anyOpenCell) {
      this.subBoardsStatus[currentSubBoard][3] = 1;
      this.subBoardsStatus[currentSubBoard][0] = 0;
      return 'cell_draw';
    }
    return 'ongoing';
  }

  setNextLegal(playedCell) {
    if (this.subBoardsStatus[playedCell][0] === 1) {
      this.subBoardsLegal.fill(0);
      this.subBoardsLegal[playedCell] = 1;
    } else {
      this.subBoardsLegal = this.subBoardsStatus.map(s => s[0]);
    }
  }

  executeMove(subBoard, cell) {
    const isLegal = this.legalCoords.some(([sb, c]) => sb === subBoard && c === cell);
    if (!isLegal) {
      throw new Error(`Attempted move ${subBoard}, ${cell}. Current turn: ${this.currentPlayer}`);
    }
    this.board[subBoard][cell][this.currentPlayer] = 1;
    this.board[subBoard][cell][0] = 0;
    let metaWin = 'ongoing';
    const win = this.checkWin(subBoard, false);
    if (win !== 'ongoing') {
      metaWin = this.checkWin(null, true);
      this.setNextLegal(cell);
    } else {
      this.setNextLegal(cell);
    }
    if (!this.gameOver) {
      this.currentPlayer = 3 - this.currentPlayer;
    }
    this.legalMoves();
    return [win, metaWin];
  }
}

function legalActionMask(game) {
  const mask = new Uint8Array(81);
  for (const [sb, c] of game.legalCoords) mask[sb * 9 + c] = 1;
  return mask;
}

return { Game, legalActionMask, WIN_LINES };

}));
