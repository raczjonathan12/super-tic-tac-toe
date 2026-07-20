const assert = require('assert');
const { Game } = require('../game.js');
const { runMctsLeafParallel, selectAction } = require('../mcts.js');

// Stub evaluator: uniform-ish policy, zero value. Matches how the Python tests use a
// freshly-initialized, untrained network -- real search, not network judgment, must find
// the forced tactic.
async function stubEvaluateFn(games) {
  const values = new Float32Array(games.length).fill(0);
  const policies = games.map(() => new Float32Array(81).fill(1 / 81));
  return { values, policies };
}

function makeImmediateWinScenario() {
  // player1 already owns sub-boards 1 and 2 (meta line [0,1,2]); sub-board 0
  // is one move from being won too, which completes the whole game.
  const game = new Game();
  game.currentPlayer = 1;
  for (let sb = 0; sb < 9; sb++) for (let c = 0; c < 9; c++) game.board[sb][c] = [1, 0, 0];
  game.board[0][0] = [0, 1, 0];
  game.board[0][1] = [0, 1, 0];
  for (let sb = 0; sb < 9; sb++) game.subBoardsStatus[sb] = [1, 0, 0, 0];
  game.subBoardsStatus[1] = [0, 1, 0, 0];
  game.subBoardsStatus[2] = [0, 1, 0, 0];
  game.subBoardsLegal.fill(1);
  game.legalMoves();
  return [game, 0 * 9 + 2];
}

function makeForcedBlockScenario() {
  // player2 already owns sub-boards 1 and 2 (meta line [0,1,2]); sub-board 0
  // is open with player2 two cells into completing it (and the whole game).
  // Every other sub-board is closed, so any non-blocking move in sub-board 0
  // sends the turn right back there.
  const game = new Game();
  game.currentPlayer = 1;
  for (let sb = 0; sb < 9; sb++) for (let c = 0; c < 9; c++) game.board[sb][c] = [1, 0, 0];
  for (let sb = 0; sb < 9; sb++) game.subBoardsStatus[sb] = [0, 0, 0, 0];
  game.subBoardsStatus[0] = [1, 0, 0, 0];
  game.subBoardsStatus[1] = [0, 0, 1, 0];
  game.subBoardsStatus[2] = [0, 0, 1, 0];
  for (let sb = 3; sb < 9; sb++) game.subBoardsStatus[sb] = [0, 0, 0, 1];
  game.board[0][0] = [0, 0, 1];
  game.board[0][1] = [0, 0, 1];
  game.subBoardsLegal.fill(0);
  game.subBoardsLegal[0] = 1;
  game.legalMoves();
  return [game, 0 * 9 + 2];
}

async function testFindsForcedWin() {
  const [game, winAction] = makeImmediateWinScenario();
  const root = await runMctsLeafParallel(game, stubEvaluateFn, 200, 8, 1.5);
  let bestAction = null, bestVisits = -1;
  for (const [action, child] of root.children) {
    if (child.visitCount > bestVisits) { bestVisits = child.visitCount; bestAction = action; }
  }
  assert.strictEqual(bestAction, winAction, `expected ${winAction}, got ${bestAction}`);
}

async function testFindsForcedBlock() {
  const [game, blockAction] = makeForcedBlockScenario();
  const root = await runMctsLeafParallel(game, stubEvaluateFn, 200, 8, 1.5);
  let bestAction = null, bestVisits = -1;
  for (const [action, child] of root.children) {
    if (child.visitCount > bestVisits) { bestVisits = child.visitCount; bestAction = action; }
  }
  assert.strictEqual(bestAction, blockAction, `expected ${blockAction}, got ${bestAction}`);
}

async function testTotalVisitsMatchNumSimulations() {
  const [game] = makeImmediateWinScenario();
  const numSimulations = 160;
  const root = await runMctsLeafParallel(game, stubEvaluateFn, numSimulations, 8, 1.5);
  let total = 0;
  for (const [, child] of root.children) total += child.visitCount;
  assert.strictEqual(total, numSimulations);
}

async function testSelectActionZeroTemperatureIsArgmaxVisits() {
  const [game] = makeImmediateWinScenario();
  const root = await runMctsLeafParallel(game, stubEvaluateFn, 50, 8, 1.5);
  const action = selectAction(root, 1e-3);
  let bestAction = null, bestVisits = -1;
  for (const [a, child] of root.children) {
    if (child.visitCount > bestVisits) { bestVisits = child.visitCount; bestAction = a; }
  }
  assert.strictEqual(action, bestAction);
}

(async () => {
  await testFindsForcedWin();
  await testFindsForcedBlock();
  await testTotalVisitsMatchNumSimulations();
  await testSelectActionZeroTemperatureIsArgmaxVisits();
  console.log('mcts.js: all tests passed');
})();
