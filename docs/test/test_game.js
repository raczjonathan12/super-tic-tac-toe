const assert = require('assert');
const { Game, legalActionMask } = require('../game.js');

function testNewGameHasNineLegalSubBoardsAllOpen() {
  const game = new Game();
  assert.strictEqual(game.gameOver, false);
  assert.strictEqual(game.legalCoords.length, 81);
  const mask = legalActionMask(game);
  assert.strictEqual(mask.reduce((a, b) => a + b, 0), 81);
}

function testExecuteMoveDoesNotFlipCurrentPlayerOnGameEndingMove() {
  // player1 owns sub-boards 1 and 2 (meta line [0,1,2]); sub-board 0 is one
  // move from being won too, which wins the whole game.
  const game = new Game();
  game.currentPlayer = 1;
  for (let sb = 0; sb < 9; sb++) {
    for (let c = 0; c < 9; c++) {
      game.board[sb][c] = [1, 0, 0];
    }
  }
  game.board[0][0] = [0, 1, 0];
  game.board[0][1] = [0, 1, 0];
  for (let sb = 0; sb < 9; sb++) game.subBoardsStatus[sb] = [1, 0, 0, 0];
  game.subBoardsStatus[1] = [0, 1, 0, 0];
  game.subBoardsStatus[2] = [0, 1, 0, 0];
  game.subBoardsLegal.fill(1);
  game.legalMoves();

  const [win, metaWin] = game.executeMove(0, 2);
  assert.strictEqual(win, 'cell_win');
  assert.strictEqual(metaWin, 'winner');
  assert.strictEqual(game.gameOver, true);
  assert.strictEqual(game.winner, 1);
  assert.strictEqual(game.currentPlayer, 1, 'currentPlayer must not flip on the winning move');
}

function testIllegalMoveThrows() {
  const game = new Game();
  const [legalSb, legalCell] = game.legalCoords[0];
  game.executeMove(legalSb, legalCell);
  assert.throws(() => game.executeMove(legalSb, legalCell));
}

testNewGameHasNineLegalSubBoardsAllOpen();
testExecuteMoveDoesNotFlipCurrentPlayerOnGameEndingMove();
testIllegalMoveThrows();
console.log('game.js: all tests passed');
