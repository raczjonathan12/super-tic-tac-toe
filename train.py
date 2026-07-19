import random
from collections import deque
import numpy as np

from network import build_network
from self_play import play_self_play_game
from evaluate import evaluate_vs_opponent


def training_step(model, batch):
    boards, statuses, legals, policy_targets, value_targets = zip(*batch)
    boards = np.stack(boards, axis=0)
    statuses = np.stack(statuses, axis=0)
    legals = np.stack(legals, axis=0)
    policy_targets = np.stack(policy_targets, axis=0)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1, 1)

    loss = model.train_on_batch(
        [boards, statuses, legals],
        {"value": value_targets, "policy": policy_targets},
    )
    return loss
