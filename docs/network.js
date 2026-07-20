(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./game.js'));
  } else {
    root.NetworkModule = factory(root.GameModule);
  }
}(typeof self !== 'undefined' ? self : this, function (GameModule) {

function encodeState(game, perspectivePlayer) {
  const mine = perspectivePlayer == null ? game.currentPlayer : perspectivePlayer;
  const opponent = 3 - mine;

  const board = [];
  for (let sb = 0; sb < 9; sb++) {
    const rows = [];
    for (let r = 0; r < 3; r++) {
      const cols = [];
      for (let c = 0; c < 3; c++) {
        const cellIdx = r * 3 + c;
        cols.push([game.board[sb][cellIdx][mine], game.board[sb][cellIdx][opponent]]);
      }
      rows.push(cols);
    }
    board.push(rows);
  }

  const status = game.subBoardsStatus.map(s => [s[0], s[mine], s[opponent], s[3]]);
  const legal = game.subBoardsLegal.slice();

  return { board, status, legal };
}

async function loadModel(modelUrl) {
  return tf.loadLayersModel(modelUrl);
}

async function predictBatch(model, games) {
  const encoded = games.map(g => encodeState(g, g.currentPlayer));
  const boardArr = encoded.map(e => e.board);
  const statusArr = encoded.map(e => e.status);
  const legalArr = encoded.map(e => e.legal);

  const boardTensor = tf.tensor(boardArr, [games.length, 9, 3, 3, 2]);
  const statusTensor = tf.tensor(statusArr, [games.length, 9, 4]);
  const legalTensor = tf.tensor(legalArr, [games.length, 9]);

  const [valueTensor, policyTensor] = model.predict([boardTensor, statusTensor, legalTensor]);
  const values = await valueTensor.data();
  const policyFlat = await policyTensor.data();

  const policies = [];
  for (let i = 0; i < games.length; i++) {
    policies.push(policyFlat.slice(i * 81, (i + 1) * 81));
  }

  boardTensor.dispose();
  statusTensor.dispose();
  legalTensor.dispose();
  valueTensor.dispose();
  policyTensor.dispose();

  return { values: Float32Array.from(values), policies };
}

return { encodeState, loadModel, predictBatch };

}));
