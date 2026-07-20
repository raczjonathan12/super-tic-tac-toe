const assert = require('assert');
const { Game } = require('../game.js');
const { encodeState } = require('../network.js');

function testEncodeStateShapesAndPerspective() {
  const game = new Game();
  game.currentPlayer = 1;
  const { board, status, legal } = encodeState(game, 1);
  assert.strictEqual(board.length, 9);
  assert.strictEqual(board[0].length, 3);
  assert.strictEqual(board[0][0].length, 3);
  assert.strictEqual(board[0][0][0].length, 2);
  assert.strictEqual(status.length, 9);
  assert.strictEqual(status[0].length, 4);
  assert.strictEqual(legal.length, 9);

  // Every cell starts open, so "mine" and "opponent" planes must be all zero.
  const anyMarked = board.some(sb => sb.some(row => row.some(cell => cell[0] === 1 || cell[1] === 1)));
  assert.strictEqual(anyMarked, false);

  // Perspective swap: encoding from player 2's perspective swaps mine/opponent.
  game.board[0][0][1] = 1;
  game.board[0][0][0] = 0;
  const asP1 = encodeState(game, 1);
  const asP2 = encodeState(game, 2);
  assert.strictEqual(asP1.board[0][0][0][0], 1); // player1's mark shows as "mine" for player1
  assert.strictEqual(asP2.board[0][0][0][1], 1); // and as "opponent" for player2
}

testEncodeStateShapesAndPerspective();
console.log('network.js: all tests passed');
