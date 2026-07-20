(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./game.js'));
  } else {
    root.MctsModule = factory(root.GameModule);
  }
}(typeof self !== 'undefined' ? self : this, function (GameModule) {

const { legalActionMask } = GameModule;

class MCTSNode {
  constructor(prior) {
    this.prior = prior;
    this.visitCount = 0;
    this.valueSum = 0.0;
    this.children = new Map();
  }
  isExpanded() { return this.children.size > 0; }
  value() { return this.visitCount === 0 ? 0.0 : this.valueSum / this.visitCount; }
}

async function batchEvaluateAndExpand(nodes, games, evaluateFn) {
  const { values, policies } = await evaluateFn(games);
  const results = [];
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const game = games[i];
    const mask = legalActionMask(game);
    const legalActions = [];
    for (let a = 0; a < 81; a++) if (mask[a]) legalActions.push(a);

    let total = 0;
    const rawProbs = legalActions.map(a => policies[i][a]);
    for (const p of rawProbs) total += p;

    legalActions.forEach((action, idx) => {
      const prob = total > 1e-8 ? rawProbs[idx] / total : 1.0 / legalActions.length;
      node.children.set(action, new MCTSNode(prob));
    });

    results.push(values[i]);
  }
  return results;
}

function selectChild(node, cPuct) {
  let bestScore = -Infinity, bestAction = null, bestChild = null;
  for (const [action, child] of node.children) {
    const q = -child.value();
    const u = cPuct * child.prior * Math.sqrt(node.visitCount) / (1 + child.visitCount);
    const score = q + u;
    if (score > bestScore) { bestScore = score; bestAction = action; bestChild = child; }
  }
  return [bestAction, bestChild];
}

function terminalOrNone(simGame) {
  if (!simGame.gameOver) return null;
  return simGame.winner != null ? -1.0 : 0.0;
}

async function runMctsLeafParallel(game, evaluateFn, numSimulations, leafBatchSize, cPuct) {
  cPuct = cPuct == null ? 1.5 : cPuct;
  const root = new MCTSNode(1.0);
  await batchEvaluateAndExpand([root], [game], evaluateFn);

  let simulationsDone = 0;
  while (simulationsDone < numSimulations) {
    const batchSize = Math.min(leafBatchSize, numSimulations - simulationsDone);

    const paths = [];
    const pendingValues = new Array(batchSize).fill(null);
    const leafPositions = [];
    const leafNodes = [];
    const leafGames = [];

    for (let i = 0; i < batchSize; i++) {
      let node = root;
      const simGame = game.clone();
      const path = [node];

      while (node.isExpanded() && !simGame.gameOver) {
        const [action, child] = selectChild(node, cPuct);
        node = child;
        node.visitCount += 1;
        node.valueSum += 1.0;
        simGame.executeMove(Math.floor(action / 9), action % 9);
        path.push(node);
      }

      paths.push(path);
      const terminalValue = terminalOrNone(simGame);
      if (terminalValue !== null) {
        pendingValues[i] = terminalValue;
      } else {
        leafPositions.push(i);
        leafNodes.push(node);
        leafGames.push(simGame);
      }
    }

    if (leafNodes.length > 0) {
      const values = await batchEvaluateAndExpand(leafNodes, leafGames, evaluateFn);
      leafPositions.forEach((pos, idx) => { pendingValues[pos] = values[idx]; });
    }

    for (let i = 0; i < batchSize; i++) {
      const path = paths[i];
      for (let j = 1; j < path.length; j++) {
        path[j].visitCount -= 1;
        path[j].valueSum -= 1.0;
      }
      let value = pendingValues[i];
      for (let j = path.length - 1; j >= 0; j--) {
        path[j].valueSum += value;
        path[j].visitCount += 1;
        value = -value;
      }
    }

    simulationsDone += batchSize;
  }

  return root;
}

function selectAction(root, temperature) {
  const actions = Array.from(root.children.keys());
  const visitCounts = actions.map(a => root.children.get(a).visitCount);
  if (temperature <= 1e-3) {
    let bestIdx = 0;
    for (let i = 1; i < visitCounts.length; i++) if (visitCounts[i] > visitCounts[bestIdx]) bestIdx = i;
    return actions[bestIdx];
  }
  const powered = visitCounts.map(v => Math.pow(v, 1 / temperature));
  const sum = powered.reduce((a, b) => a + b, 0);
  const probs = powered.map(p => p / sum);
  let r = Math.random(), acc = 0;
  for (let i = 0; i < actions.length; i++) {
    acc += probs[i];
    if (r <= acc) return actions[i];
  }
  return actions[actions.length - 1];
}

return { MCTSNode, batchEvaluateAndExpand, selectChild, terminalOrNone, runMctsLeafParallel, selectAction };

}));
