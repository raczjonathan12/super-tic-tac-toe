import math
import numpy as np
from game import legal_action_mask
from network import encode_state


class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def evaluate_and_expand(node, game, model):
    board, status, legal = encode_state(game, game.current_player)
    board_b = np.expand_dims(board, 0)
    status_b = np.expand_dims(status, 0)
    legal_b = np.expand_dims(legal, 0)

    value_out, policy_out = model([board_b, status_b, legal_b])
    value = float(value_out.numpy()[0, 0])
    policy = policy_out.numpy()[0]

    mask = legal_action_mask(game)
    legal_actions = np.nonzero(mask)[0]
    legal_probs = policy[legal_actions]
    total = legal_probs.sum()
    if total > 1e-8:
        legal_probs = legal_probs / total
    else:
        legal_probs = np.full(len(legal_actions), 1.0 / len(legal_actions))

    for action, prob in zip(legal_actions, legal_probs):
        node.children[int(action)] = MCTSNode(prior=float(prob))

    return value


def select_child(node, c_puct):
    best_score = -float("inf")
    best_action = None
    best_child = None
    for action, child in node.children.items():
        q = -child.value()
        u = c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def run_mcts(game, model, num_simulations, c_puct=1.5, dirichlet_alpha=0.3,
             dirichlet_epsilon=0.25, add_noise=False):
    root = MCTSNode(prior=1.0)
    evaluate_and_expand(root, game, model)

    if add_noise and len(root.children) > 0:
        actions = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1 - dirichlet_epsilon) + n * dirichlet_epsilon

    for _ in range(num_simulations):
        node = root
        sim_game = game.clone()
        path = [node]

        while node.is_expanded() and not sim_game.game_over:
            action, node = select_child(node, c_puct)
            sub_board, cell = action // 9, action % 9
            sim_game.execute_move(sub_board, cell)
            path.append(node)

        if sim_game.game_over:
            # execute_move does not flip current_player on a game-ending move,
            # so sim_game.current_player is the winner here, not "whoever
            # moves next". Value must be from the perspective of whoever's
            # turn it would conceptually be (the loser), to stay consistent
            # with the non-terminal case and the alternating-sign backup.
            value = -1.0 if sim_game.winner is not None else 0.0
        else:
            value = evaluate_and_expand(node, sim_game, model)

        for path_node in reversed(path):
            path_node.value_sum += value
            path_node.visit_count += 1
            value = -value

    return root


def get_policy_target(root):
    target = np.zeros(81, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits == 0:
        for action, child in root.children.items():
            target[action] = child.prior
    else:
        for action, child in root.children.items():
            target[action] = child.visit_count / total_visits
    return target


def select_action(root, temperature):
    actions = list(root.children.keys())
    visit_counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
    if temperature <= 1e-3:
        best_idx = int(np.argmax(visit_counts))
        return actions[best_idx]
    powered = visit_counts ** (1.0 / temperature)
    probs = powered / powered.sum()
    return int(np.random.choice(actions, p=probs))
