"""Supervised RNN and reservoir navigation benchmark for GeoMem.

This script converts the current generated geometries into goal-conditioned
navigation imitation tasks. It intentionally avoids full RL: labels are
shortest-path optimal actions, so failures are easier to attribute to memory
architecture rather than exploration.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Callable, Hashable

import numpy as np

try:
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    MLPClassifier = None

    def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
        return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

from commutativity_probe import (
    Environment,
    State,
    bottleneck_rooms,
    clamped_lattice,
    torus_lattice,
    trace_monoid,
    tree,
)
from plot_current_results import esc
from policy_ambiguity import navigation_graph


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

OBS_MODES = ("vector", "sparse_aliased", "egocentric_landmark", "hybrid")
TEMPORAL_MODES = ("temporal_full", "temporal_dropout", "temporal_cue_then_blank")
TEMPORAL_DROPOUT_P = 0.5
DATASET_SEEDS = (11, 17)
RESERVOIR_SEEDS = (31, 37, 41)
HIDDEN_DIM = 32
ROLLOUT_LIMIT = 20
LIN_SEQUENCE_CONFIGS = tuple((r, length) for r in (1, 3, 5) for length in (2, 4, 8))
LIN_SEQUENCE_REFERENCE = (3, 4)
LIN_LONG_SEQUENCE_CONFIGS = tuple((r, length) for r in (1, 3, 5) for length in (4, 16, 64))
LIN_LONG_DATASET_SEEDS = (11, 17, 23)
LONG_HISTORY_LENGTHS = (4, 16, 64, 80)
RESERVOIR_KINDS = (
    "random",
    "symmetric",
    "oscillatory",
    "orthogonal",
    "nilpotent",
    "diagonal",
    "low_rank",
    "sparse",
    "block_hybrid",
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env: Environment
    root: State
    geometry: str
    behavior_family: str = "formal"
    max_examples: int = 650
    n_goals: int = 3


@dataclass
class Dataset:
    task: TaskSpec
    obs_mode: str
    temporal_mode: str
    temporal_dropout_p: float
    action_labels: list[str]
    x_train: list[np.ndarray]
    y_train: list[int]
    state_train: list[State]
    goal_train: list[State]
    x_test: list[np.ndarray]
    y_test: list[int]
    state_test: list[State]
    goal_test: list[State]
    feature_dim: int


def stable_seed(*parts: object) -> int:
    text = "|".join(repr(part) for part in parts)
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def aliased_rooms(size: int = 3) -> Environment:
    """Two identical rooms with aliased local observations and a bottleneck.

    The latent state includes room id, but sparse observations deliberately hide
    the room. This creates a small navigation-like POMDP where current local
    appearance is not enough to know which room the agent occupies.
    """
    actions = ("N", "S", "E", "W")
    states = tuple((room, x, y) for room in range(2) for x in range(size) for y in range(size))
    state_set = set(states)
    transition: dict[tuple[State, str], State] = {}
    valid: set[tuple[State, str]] = set()
    for room, x, y in states:
        candidates = {
            "N": (room, x, min(size - 1, y + 1)),
            "S": (room, x, max(0, y - 1)),
            "E": (room, min(size - 1, x + 1), y),
            "W": (room, max(0, x - 1), y),
        }
        for action, target in candidates.items():
            transition[((room, x, y), action)] = target
            if target != (room, x, y):
                valid.add(((room, x, y), action))
    left_door = (0, size - 1, size // 2)
    right_door = (1, 0, size // 2)
    transition[(left_door, "E")] = right_door
    transition[(right_door, "W")] = left_door
    valid.add((left_door, "E"))
    valid.add((right_door, "W"))
    return Environment("aliased_rooms", states, actions, transition, frozenset(valid))


def aliased_t_maze(corridor_length: int = 8) -> Environment:
    """Cue-dependent T-maze with aliased corridor observations.

    The latent state stores which cue branch was entered from the root. Sparse
    observations hide that cue after the first step, so the correct arm at the
    junction depends on remembered history.
    """
    actions = ("C0", "C1", "F", "B", "L", "R")
    left_arm = corridor_length
    right_arm = corridor_length + 1
    states: list[State] = [()]
    for cue in (0, 1):
        for pos in range(corridor_length + 2):
            states.append((cue, pos))
    transition: dict[tuple[State, str], State] = {}
    valid: set[tuple[State, str]] = set()
    for state in states:
        for action in actions:
            transition[(state, action)] = state
    transition[((), "C0")] = (0, 0)
    transition[((), "C1")] = (1, 0)
    valid.add(((), "C0"))
    valid.add(((), "C1"))
    for cue in (0, 1):
        for pos in range(corridor_length - 1):
            transition[((cue, pos), "F")] = (cue, pos + 1)
            valid.add(((cue, pos), "F"))
        for pos in range(1, corridor_length):
            transition[((cue, pos), "B")] = (cue, pos - 1)
            valid.add(((cue, pos), "B"))
        transition[((cue, corridor_length - 1), "L")] = (cue, left_arm)
        transition[((cue, corridor_length - 1), "R")] = (cue, right_arm)
        transition[((cue, left_arm), "B")] = (cue, corridor_length - 1)
        transition[((cue, right_arm), "B")] = (cue, corridor_length - 1)
        valid.update({
            ((cue, corridor_length - 1), "L"),
            ((cue, corridor_length - 1), "R"),
            ((cue, left_arm), "B"),
            ((cue, right_arm), "B"),
        })
    return Environment("aliased_t_maze", tuple(states), actions, transition, frozenset(valid))


def landmark_gap_detour() -> Environment:
    """Cue, sensory gap, forced detour, then route-dependent choice."""
    actions = ("C0", "C1", "F", "UP", "DOWN", "B", "L", "R")
    states: list[State] = [()]
    for cue in (0, 1):
        for stage in range(9):
            states.append((cue, stage))
    transition: dict[tuple[State, str], State] = {}
    valid: set[tuple[State, str]] = set()
    for state in states:
        for action in actions:
            transition[(state, action)] = state
    transition[((), "C0")] = (0, 0)
    transition[((), "C1")] = (1, 0)
    valid.update({((), "C0"), ((), "C1")})
    progress = {0: "F", 1: "F", 2: "UP", 3: "DOWN", 4: "F", 5: "F"}
    for cue in (0, 1):
        for stage, action in progress.items():
            transition[((cue, stage), action)] = (cue, stage + 1)
            transition[((cue, stage + 1), "B")] = (cue, stage)
            valid.add(((cue, stage), action))
            valid.add(((cue, stage + 1), "B"))
        transition[((cue, 6), "L")] = (cue, 7)
        transition[((cue, 6), "R")] = (cue, 8)
        transition[((cue, 7), "B")] = (cue, 6)
        transition[((cue, 8), "B")] = (cue, 6)
        valid.update({((cue, 6), "L"), ((cue, 6), "R"), ((cue, 7), "B"), ((cue, 8), "B")})
    return Environment("landmark_gap_detour", tuple(states), actions, transition, frozenset(valid))


def shortcut_route_choice() -> Environment:
    """Identical final junction after short or turning route histories."""
    actions = ("Q0", "Q1", "F", "TURN", "B", "CUT", "ARC")
    states: list[State] = [()]
    for route in (0, 1):
        for stage in range(8):
            states.append((route, stage))
    transition: dict[tuple[State, str], State] = {}
    valid: set[tuple[State, str]] = set()
    for state in states:
        for action in actions:
            transition[(state, action)] = state
    transition[((), "Q0")] = (0, 0)
    transition[((), "Q1")] = (1, 0)
    valid.update({((), "Q0"), ((), "Q1")})
    for route in (0, 1):
        for stage in range(5):
            action = "TURN" if route == 1 and stage in {1, 3} else "F"
            transition[((route, stage), action)] = (route, stage + 1)
            transition[((route, stage + 1), "B")] = (route, stage)
            valid.add(((route, stage), action))
            valid.add(((route, stage + 1), "B"))
        transition[((route, 5), "CUT")] = (route, 6)
        transition[((route, 5), "ARC")] = (route, 7)
        transition[((route, 6), "B")] = (route, 5)
        transition[((route, 7), "B")] = (route, 5)
        valid.update({((route, 5), "CUT"), ((route, 5), "ARC"), ((route, 6), "B"), ((route, 7), "B")})
    return Environment("shortcut_route_choice", tuple(states), actions, transition, frozenset(valid))


def aliased_loop_rooms(size: int = 2, rooms: int = 3) -> Environment:
    """Ring of identical small rooms with hidden room identity."""
    actions = ("N", "S", "E", "W")
    states = tuple((room, x, y) for room in range(rooms) for x in range(size) for y in range(size))
    transition: dict[tuple[State, str], State] = {}
    valid: set[tuple[State, str]] = set()
    for room, x, y in states:
        candidates = {
            "N": (room, x, min(size - 1, y + 1)),
            "S": (room, x, max(0, y - 1)),
            "E": (room, min(size - 1, x + 1), y),
            "W": (room, max(0, x - 1), y),
        }
        for action, target in candidates.items():
            transition[((room, x, y), action)] = target
            if target != (room, x, y):
                valid.add(((room, x, y), action))
    for room in range(rooms):
        right_room = (room + 1) % rooms
        left_room = (room - 1) % rooms
        east_door = (room, size - 1, 0)
        west_door = (room, 0, size - 1)
        transition[(east_door, "E")] = (right_room, 0, 0)
        transition[(west_door, "W")] = (left_room, size - 1, size - 1)
        valid.add((east_door, "E"))
        valid.add((west_door, "W"))
    return Environment("aliased_loop_rooms", states, actions, transition, frozenset(valid))


def trace_env(commuting: frozenset[tuple[str, str]], name_suffix: str) -> Environment:
    env = trace_monoid(commuting, max_depth=6)
    return Environment(f"trace_{name_suffix}", env.states, env.actions, env.transition, env.valid, env.max_depth)


def task_specs() -> list[TaskSpec]:
    return [
        TaskSpec("torus_lattice", torus_lattice(width=4, height=4), (0, 0), "metric_commutative", "metric_landmark", max_examples=180),
        TaskSpec("bottleneck_rooms", bottleneck_rooms(size=3), (0, 0, 0), "local_metric_global_graph", "bottleneck_route", max_examples=220),
        TaskSpec("tree", tree(branching=3, depth=4), (), "tree_noncommutative", max_examples=260),
        TaskSpec("aliased_rooms", aliased_rooms(size=3), (0, 0, 0), "aliased_navigation", "repeated_room", max_examples=220),
        TaskSpec("aliased_t_maze", aliased_t_maze(), (), "cue_dependent_aliased_navigation", "delayed_landmark_choice", max_examples=180, n_goals=2),
        TaskSpec("landmark_gap_detour", landmark_gap_detour(), (), "delayed_gap_navigation", "landmark_gap_detour", max_examples=180, n_goals=2),
        TaskSpec("shortcut_route_choice", shortcut_route_choice(), (), "shortcut_route_navigation", "shortcut_route_memory", max_examples=180, n_goals=2),
        TaskSpec("aliased_loop_rooms", aliased_loop_rooms(), (0, 0, 0), "looped_aliased_navigation", "repeated_room_loop", max_examples=220),
        TaskSpec("trace_commute_000", trace_env(frozenset(), "commute_000"), (), "trace_noncommutative", max_examples=320),
        TaskSpec("trace_commute_067", trace_env(frozenset({("A", "B"), ("A", "C")}), "commute_067"), (), "trace_partial", max_examples=320),
        TaskSpec("trace_commute_100", trace_env(frozenset({("A", "B"), ("A", "C"), ("B", "C")}), "commute_100"), (), "trace_commutative", max_examples=320),
    ]


def long_behavior_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(
            "long_aliased_t_maze",
            aliased_t_maze(corridor_length=72),
            (),
            "long_cue_dependent_navigation",
            "long_delayed_landmark_choice",
            max_examples=420,
            n_goals=2,
        ),
    ]


def graph_distances(graph: dict[State, list[tuple[str, State]]], source: State) -> tuple[dict[State, int], dict[State, State]]:
    distances = {source: 0}
    parent: dict[State, State] = {}
    queue: deque[State] = deque([source])
    while queue:
        state = queue.popleft()
        for _, neighbor in graph[state]:
            if neighbor not in distances:
                distances[neighbor] = distances[state] + 1
                parent[neighbor] = state
                queue.append(neighbor)
    return distances, parent


def path_from_parent(root: State, state: State, parent: dict[State, State]) -> list[State]:
    if state == root:
        return [root]
    if state not in parent:
        return [root]
    path = [state]
    while path[-1] != root:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def select_goals(task: TaskSpec, graph: dict[State, list[tuple[str, State]]]) -> list[State]:
    if task.name in {"aliased_t_maze", "long_aliased_t_maze"}:
        arm_states = sorted(state for state in task.env.states if state != () and state[1] >= 2)
        left_arm = max(state[1] for state in arm_states) - 1
        right_arm = left_arm + 1
        return [(0, left_arm), (1, right_arm)]
    if task.name == "landmark_gap_detour":
        return [(0, 7), (1, 8)]
    if task.name == "shortcut_route_choice":
        return [(0, 6), (1, 7)]
    distances, _ = graph_distances(graph, task.root)
    reachable = sorted((state for state in task.env.states if state in distances), key=lambda s: (distances[s], s))
    non_root = [state for state in reachable if state != task.root]
    if len(non_root) <= task.n_goals:
        return non_root
    indices = np.linspace(len(non_root) // 2, len(non_root) - 1, task.n_goals, dtype=int)
    return [non_root[int(idx)] for idx in indices]


def state_numeric(state: State, width: int = 6) -> np.ndarray:
    values = [float(value) for value in state[:width]]
    values += [0.0] * (width - len(values))
    scale = max(1.0, max((abs(v) for v in values), default=1.0))
    return np.asarray(values, dtype=np.float32) / scale


def action_counts(state: State, n_actions: int = 3) -> np.ndarray:
    counts = np.zeros(n_actions, dtype=np.float32)
    for value in state:
        if isinstance(value, int) and 0 <= value < n_actions:
            counts[value] += 1.0
    denom = max(1.0, float(len(state)))
    return counts / denom


def sparse_symbol(state: State, task: TaskSpec) -> int:
    if task.name.startswith("trace"):
        return (state[-1] if state else 3) % 4
    if task.name == "aliased_rooms":
        _, x, y = state
        return (x + 2 * y) % 5
    if task.name == "aliased_loop_rooms":
        _, x, y = state
        return (x + 2 * y) % 5
    if task.name.startswith("room_graph_"):
        _, x, y = state
        return (x + 2 * y) % 5
    if task.name in {"aliased_t_maze", "long_aliased_t_maze"}:
        if state == ():
            return 0
        cue, pos = state
        if pos == 0:
            return 1 + cue
        max_pos = max((candidate[1] for candidate in task.env.states if candidate), default=9)
        if pos >= max_pos - 1:
            return 5
        return 4
    if task.name == "landmark_gap_detour":
        if state == ():
            return 0
        cue, stage = state
        if stage == 0:
            return 1 + cue
        if stage in {2, 3, 4}:
            return 4
        if stage >= 6:
            return 5
        return 3
    if task.name == "shortcut_route_choice":
        if state == ():
            return 0
        route, stage = state
        if stage == 0:
            return 1 + route
        if stage == 5:
            return 5
        return 4
    if task.name in {"torus_lattice", "clamped_lattice"}:
        x, y = state
        return (x + y) % 4
    if task.name == "bottleneck_rooms":
        _, x, y = state
        return (x + y) % 4
    return (len(state), state[-1] if state else 0)[-1] % 4


def one_hot(index: int, size: int) -> np.ndarray:
    values = np.zeros(size, dtype=np.float32)
    values[index % size] = 1.0
    return values


def state_feature(state: State, task: TaskSpec, mode: str, state_index: dict[State, int]) -> np.ndarray:
    if mode == "dense":
        return one_hot(state_index[state], len(state_index))
    if mode == "vector":
        if task.name.startswith("trace"):
            return action_counts(state, n_actions=3)
        return state_numeric(state, width=4)
    if mode == "sparse_aliased":
        return one_hot(sparse_symbol(state, task), 6)
    if mode == "egocentric_landmark":
        action_bits = np.asarray(
            [float((state, action) in task.env.valid) for action in task.env.actions],
            dtype=np.float32,
        )
        return np.concatenate([action_bits, one_hot(sparse_symbol(state, task), 6)])
    if mode == "hybrid":
        return np.concatenate([
            state_feature(state, task, "vector", state_index),
            state_feature(state, task, "sparse_aliased", state_index),
        ])
    raise ValueError(f"unknown observation mode: {mode}")


def temporal_keep_mask(length: int, temporal_mode: str, dropout_p: float, seed: int) -> np.ndarray:
    if temporal_mode == "temporal_full":
        return np.ones(length, dtype=bool)
    if temporal_mode == "temporal_dropout":
        rng = np.random.default_rng(seed)
        return rng.random(length) >= dropout_p
    if temporal_mode == "temporal_landmark":
        mask = np.zeros(length, dtype=bool)
        mask[::3] = True
        if length:
            mask[-1] = True
        return mask
    if temporal_mode == "temporal_cue_then_blank":
        mask = np.zeros(length, dtype=bool)
        mask[:min(2, length)] = True
        return mask
    raise ValueError(f"unknown temporal mode: {temporal_mode}")


def sequence_features(
    path: list[State],
    goal: State,
    task: TaskSpec,
    mode: str,
    state_index: dict[State, int],
    temporal_mode: str = "temporal_full",
    temporal_dropout_p: float = TEMPORAL_DROPOUT_P,
    temporal_seed: int = 0,
) -> np.ndarray:
    goal_feature = state_feature(goal, task, mode, state_index)
    state_features = [state_feature(state, task, mode, state_index) for state in path]
    keep = temporal_keep_mask(len(path), temporal_mode, temporal_dropout_p, temporal_seed)
    frames = []
    for idx, feature in enumerate(state_features):
        visible_feature = feature if keep[idx] else np.zeros_like(feature)
        frames.append(np.concatenate([visible_feature, goal_feature]))
    return np.asarray(frames, dtype=np.float32)


def stratified_example_split(
    examples: list[tuple[np.ndarray, int, State, State]],
    rng: Random,
) -> tuple[list[tuple[np.ndarray, int, State, State]], list[tuple[np.ndarray, int, State, State]]]:
    by_label: dict[int, list[tuple[np.ndarray, int, State, State]]] = defaultdict(list)
    for example in examples:
        by_label[example[1]].append(example)
    train: list[tuple[np.ndarray, int, State, State]] = []
    test: list[tuple[np.ndarray, int, State, State]] = []
    for group in by_label.values():
        rng.shuffle(group)
        test_count = max(1, int(round(0.3 * len(group)))) if len(group) > 1 else 0
        test.extend(group[:test_count])
        train.extend(group[test_count:] or group[:1])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test or examples[:1]


def build_tmaze_dataset(
    task: TaskSpec,
    obs_mode: str,
    temporal_mode: str,
    temporal_dropout_p: float,
    seed: int,
) -> Dataset:
    """Directed cue-corridor task where the sparse goal is intentionally aliased."""
    rng = Random(seed)
    state_index = {state: idx for idx, state in enumerate(sorted(task.env.states))}
    action_labels = sorted(task.env.actions)
    action_index = {action: idx for idx, action in enumerate(action_labels)}
    max_pos = max(state[1] for state in task.env.states if state != ())
    left_arm = max_pos - 1
    right_arm = max_pos
    junction = left_arm - 1

    examples: list[tuple[np.ndarray, int, State, State]] = []
    for cue, goal, turn_action in ((0, (0, left_arm), "L"), (1, (1, right_arm), "R")):
        for pos in range(junction + 1):
            state = (cue, pos)
            path: list[State] = [()] + [(cue, step) for step in range(pos + 1)]
            action = "F" if pos < junction else turn_action
            decision_repeats = junction if task.behavior_family == "long_delayed_landmark_choice" else 4
            repeats = decision_repeats if pos == junction else 1
            for variant in range(repeats):
                temporal_seed = stable_seed(seed, task.name, obs_mode, temporal_mode, goal, state, variant)
                examples.append((
                    sequence_features(path, goal, task, obs_mode, state_index, temporal_mode, temporal_dropout_p, temporal_seed),
                    action_index[action],
                    state,
                    goal,
                ))

    train, test = stratified_example_split(examples, rng)
    return Dataset(
        task=task,
        obs_mode=obs_mode,
        temporal_mode=temporal_mode,
        temporal_dropout_p=temporal_dropout_p,
        action_labels=action_labels,
        x_train=[item[0] for item in train],
        y_train=[item[1] for item in train],
        state_train=[item[2] for item in train],
        goal_train=[item[3] for item in train],
        x_test=[item[0] for item in test],
        y_test=[item[1] for item in test],
        state_test=[item[2] for item in test],
        goal_test=[item[3] for item in test],
        feature_dim=train[0][0].shape[1],
    )


def build_choice_route_dataset(
    task: TaskSpec,
    obs_mode: str,
    temporal_mode: str,
    temporal_dropout_p: float,
    seed: int,
) -> Dataset:
    """Directed route-memory tasks with delayed context-dependent choices."""
    rng = Random(seed)
    state_index = {state: idx for idx, state in enumerate(sorted(task.env.states))}
    action_labels = sorted(task.env.actions)
    action_index = {action: idx for idx, action in enumerate(action_labels)}
    if task.name == "landmark_gap_detour":
        stage_actions = ("F", "F", "UP", "DOWN", "F", "F")
        goal_turns = ((0, (0, 7), "L"), (1, (1, 8), "R"))
        decision_stage = 6
    elif task.name == "shortcut_route_choice":
        stage_actions = ("F", "F", "F", "F", "F")
        goal_turns = ((0, (0, 6), "CUT"), (1, (1, 7), "ARC"))
        decision_stage = 5
    else:
        raise ValueError(f"unsupported route choice task: {task.name}")

    examples: list[tuple[np.ndarray, int, State, State]] = []
    for context, goal, turn_action in goal_turns:
        for stage in range(decision_stage + 1):
            state = (context, stage)
            path: list[State] = [()] + [(context, step) for step in range(stage + 1)]
            action = stage_actions[stage] if stage < len(stage_actions) else turn_action
            repeats = 4 if stage == decision_stage else 1
            for variant in range(repeats):
                temporal_seed = stable_seed(seed, task.name, obs_mode, temporal_mode, goal, state, variant)
                examples.append((
                    sequence_features(path, goal, task, obs_mode, state_index, temporal_mode, temporal_dropout_p, temporal_seed),
                    action_index[action],
                    state,
                    goal,
                ))

    train, test = stratified_example_split(examples, rng)
    return Dataset(
        task=task,
        obs_mode=obs_mode,
        temporal_mode=temporal_mode,
        temporal_dropout_p=temporal_dropout_p,
        action_labels=action_labels,
        x_train=[item[0] for item in train],
        y_train=[item[1] for item in train],
        state_train=[item[2] for item in train],
        goal_train=[item[3] for item in train],
        x_test=[item[0] for item in test],
        y_test=[item[1] for item in test],
        state_test=[item[2] for item in test],
        goal_test=[item[3] for item in test],
        feature_dim=train[0][0].shape[1],
    )


def shortest_policy(graph: dict[State, list[tuple[str, State]]], goal: State) -> dict[State, frozenset[str]]:
    distances, _ = graph_distances(graph, goal)
    policies: dict[State, frozenset[str]] = {}
    for state, distance in distances.items():
        if distance == 0:
            continue
        optimal = frozenset(
            action for action, neighbor in graph[state]
            if neighbor in distances and distances[neighbor] == distance - 1
        )
        if optimal:
            policies[state] = optimal
    return policies


def build_dataset(
    task: TaskSpec,
    obs_mode: str,
    temporal_mode: str = "temporal_full",
    temporal_dropout_p: float = TEMPORAL_DROPOUT_P,
    seed: int = 7,
) -> Dataset:
    if task.name in {"aliased_t_maze", "long_aliased_t_maze"}:
        return build_tmaze_dataset(task, obs_mode, temporal_mode, temporal_dropout_p, seed)
    if task.name in {"landmark_gap_detour", "shortcut_route_choice"}:
        return build_choice_route_dataset(task, obs_mode, temporal_mode, temporal_dropout_p, seed)
    rng = Random(seed)
    graph = navigation_graph(task.env)
    _, parent = graph_distances(graph, task.root)
    state_index = {state: idx for idx, state in enumerate(sorted(task.env.states))}
    goals = select_goals(task, graph)
    action_labels = sorted({action for edges in graph.values() for action, _ in edges})
    action_index = {action: idx for idx, action in enumerate(action_labels)}

    examples: list[tuple[np.ndarray, int, State, State]] = []
    for goal in goals:
        policies = shortest_policy(graph, goal)
        for state, policy in policies.items():
            if state == goal:
                continue
            if task.name in {"aliased_t_maze", "long_aliased_t_maze"} and state == task.root:
                continue
            path = path_from_parent(task.root, state, parent)
            if not path:
                continue
            action = sorted(policy)[0]
            temporal_seed = stable_seed(seed, task.name, obs_mode, temporal_mode, goal, state)
            examples.append((
                sequence_features(path, goal, task, obs_mode, state_index, temporal_mode, temporal_dropout_p, temporal_seed),
                action_index[action],
                state,
                goal,
            ))
    rng.shuffle(examples)
    examples = examples[:task.max_examples]
    split = max(1, int(round(0.7 * len(examples))))
    train = examples[:split]
    test = examples[split:] or examples[:1]
    return Dataset(
        task=task,
        obs_mode=obs_mode,
        temporal_mode=temporal_mode,
        temporal_dropout_p=temporal_dropout_p,
        action_labels=action_labels,
        x_train=[item[0] for item in train],
        y_train=[item[1] for item in train],
        state_train=[item[2] for item in train],
        goal_train=[item[3] for item in train],
        x_test=[item[0] for item in test],
        y_test=[item[1] for item in test],
        state_test=[item[2] for item in test],
        goal_test=[item[3] for item in test],
        feature_dim=train[0][0].shape[1],
    )


def pad_last_features(xs: list[np.ndarray]) -> np.ndarray:
    return np.asarray([x[-1] for x in xs], dtype=np.float32)


def pad_history_features(xs: list[np.ndarray], history_len: int) -> np.ndarray:
    feature_dim = xs[0].shape[1]
    features = np.zeros((len(xs), history_len, feature_dim), dtype=np.float32)
    for idx, x in enumerate(xs):
        visible = min(history_len, len(x))
        features[idx, -visible:] = x[-visible:]
    return features.reshape(len(xs), history_len * feature_dim)


def behavioral_context_targets(dataset: Dataset) -> tuple[list[int], list[int]] | None:
    context_families = {
        "repeated_room",
        "repeated_room_loop",
        "delayed_landmark_choice",
        "long_delayed_landmark_choice",
        "landmark_gap_detour",
        "shortcut_route_memory",
    }
    if dataset.task.behavior_family not in context_families:
        return None
    train = [int(state[0]) for state in dataset.state_train if state]
    test = [int(state[0]) for state in dataset.state_test if state]
    if len(train) != len(dataset.state_train) or len(test) != len(dataset.state_test):
        return None
    if len(set(train)) < 2:
        return None
    return train, test


def train_linear_classifier(x_train: np.ndarray, y_train: list[int], x_test: np.ndarray) -> np.ndarray:
    return fit_linear_predictor(x_train, y_train)(x_test)


def fit_linear_predictor(x_train: np.ndarray, y_train: list[int]) -> Callable[[np.ndarray], np.ndarray]:
    classes = sorted(set(y_train))
    if len(classes) > 1:
        x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
        y_onehot = np.zeros((x_train.shape[0], len(classes)), dtype=np.float32)
        class_to_col = {label: idx for idx, label in enumerate(classes)}
        class_array = np.asarray(classes, dtype=int)
        for idx, label in enumerate(y_train):
            y_onehot[idx, class_to_col[label]] = 1.0
        if x_aug.shape[1] <= x_aug.shape[0]:
            reg = 1e-3 * np.eye(x_aug.shape[1], dtype=np.float32)
            weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_onehot)
        else:
            reg = 1e-3 * np.eye(x_aug.shape[0], dtype=np.float32)
            dual = np.linalg.solve(x_aug @ x_aug.T + reg, y_onehot)
            weights = x_aug.T @ dual

        def predict(x: np.ndarray) -> np.ndarray:
            x_eval = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
            return class_array[np.argmax(x_eval @ weights, axis=1)].astype(int)

        return predict

    means = {label: x_train[[idx for idx, y in enumerate(y_train) if y == label]].mean(axis=0) for label in classes}

    def predict(x: np.ndarray) -> np.ndarray:
        preds = []
        for row in x:
            preds.append(min(classes, key=lambda label: float(np.linalg.norm(row - means[label]))))
        return np.asarray(preds, dtype=int)

    return predict


def fit_mlp_predictor(x_train: np.ndarray, y_train: list[int], seed: int) -> Callable[[np.ndarray], np.ndarray]:
    if MLPClassifier is not None and len(set(y_train)) > 1:
        clf = MLPClassifier(
            hidden_layer_sizes=(24,),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=0.01,
            max_iter=140,
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=8,
        )
        clf.fit(x_train, y_train)
        return lambda x: clf.predict(x).astype(int)
    return fit_linear_predictor(x_train, y_train)


def context_readout_accuracy(
    x_train: np.ndarray,
    x_test: np.ndarray,
    context_targets: tuple[list[int], list[int]] | None,
) -> float:
    if context_targets is None:
        return float("nan")
    context_train, context_test = context_targets
    context_predictor = fit_linear_predictor(x_train, context_train)
    return accuracy_score(context_test, list(context_predictor(x_test)))


class NumpyRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 0.35 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
        self.w_rec = rng.normal(0.0, 0.45 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.w_out = rng.normal(0.0, 0.35 / math.sqrt(hidden_dim), size=(output_dim, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        h = np.zeros(self.w_rec.shape[0], dtype=np.float32)
        states = [h.copy()]
        for frame in x:
            h = np.tanh(self.w_in @ frame + self.w_rec @ h + self.b_h)
            states.append(h.copy())
        logits = self.w_out @ h + self.b_out
        return states, logits

    def predict(self, xs: list[np.ndarray]) -> np.ndarray:
        return np.asarray([int(np.argmax(self.forward(x)[1])) for x in xs], dtype=int)

    def final_states(self, xs: list[np.ndarray]) -> np.ndarray:
        return np.asarray([self.forward(x)[0][-1] for x in xs], dtype=np.float32)

    def fit(self, xs: list[np.ndarray], ys: list[int], *, epochs: int = 14, lr: float = 0.015, seed: int = 1) -> list[float]:
        rng = Random(seed)
        losses: list[float] = []
        order = list(range(len(xs)))
        for _ in range(epochs):
            rng.shuffle(order)
            total_loss = 0.0
            for idx in order:
                x = xs[idx]
                y = ys[idx]
                states, logits = self.forward(x)
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / np.sum(probs)
                total_loss += float(-np.log(max(1e-8, probs[y])))
                d_logits = probs
                d_logits[y] -= 1.0
                grad_w_out = np.outer(d_logits, states[-1])
                grad_b_out = d_logits
                dh = self.w_out.T @ d_logits
                grad_w_in = np.zeros_like(self.w_in)
                grad_w_rec = np.zeros_like(self.w_rec)
                grad_b_h = np.zeros_like(self.b_h)
                for t in range(len(x), 0, -1):
                    h = states[t]
                    h_prev = states[t - 1]
                    dz = dh * (1.0 - h * h)
                    grad_w_in += np.outer(dz, x[t - 1])
                    grad_w_rec += np.outer(dz, h_prev)
                    grad_b_h += dz
                    dh = self.w_rec.T @ dz
                for grad in (grad_w_in, grad_w_rec, grad_b_h, grad_w_out, grad_b_out):
                    np.clip(grad, -1.0, 1.0, out=grad)
                self.w_in -= lr * grad_w_in
                self.w_rec -= lr * grad_w_rec
                self.b_h -= lr * grad_b_h
                self.w_out -= lr * grad_w_out
                self.b_out -= lr * grad_b_out
            losses.append(total_loss / max(1, len(xs)))
        return losses


def normalize_spectral_radius(matrix: np.ndarray, radius: float = 0.9) -> np.ndarray:
    eigvals = np.linalg.eigvals(matrix)
    current = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    if current < 1e-8:
        return matrix
    return (matrix * (radius / current)).astype(np.float32)


def make_reservoir_matrix(kind: str, hidden_dim: int, seed: int = 0, radius: float = 0.9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        matrix = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
    elif kind == "symmetric":
        base = rng.normal(size=(hidden_dim, hidden_dim))
        matrix = (base + base.T) / 2.0
    elif kind == "oscillatory":
        base = rng.normal(size=(hidden_dim, hidden_dim))
        matrix = base - base.T
    elif kind == "orthogonal":
        q, _ = np.linalg.qr(rng.normal(size=(hidden_dim, hidden_dim)))
        matrix = q
    elif kind == "nilpotent":
        matrix = np.zeros((hidden_dim, hidden_dim))
        for idx in range(hidden_dim - 1):
            matrix[idx + 1, idx] = 1.0
    elif kind == "diagonal":
        matrix = np.diag(np.linspace(0.15, radius, hidden_dim))
        return matrix.astype(np.float32)
    elif kind == "low_rank":
        u = rng.normal(size=(hidden_dim, 4))
        v = rng.normal(size=(4, hidden_dim))
        matrix = u @ v / math.sqrt(hidden_dim)
    elif kind == "sparse":
        mask = rng.random((hidden_dim, hidden_dim)) < 0.12
        matrix = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)) * mask
    elif kind == "block_hybrid":
        matrix = np.zeros((hidden_dim, hidden_dim))
        third = hidden_dim // 3
        matrix[:third, :third] = make_reservoir_matrix("symmetric", third, seed + 1, radius=0.7)
        matrix[third:2 * third, third:2 * third] = make_reservoir_matrix("oscillatory", third, seed + 2, radius=0.9)
        tail = hidden_dim - 2 * third
        matrix[2 * third:, 2 * third:] = make_reservoir_matrix("nilpotent", tail, seed + 3, radius=0.9)
    else:
        raise ValueError(f"unknown reservoir kind: {kind}")
    return normalize_spectral_radius(np.asarray(matrix, dtype=np.float32), radius=radius)


def reservoir_states(xs: list[np.ndarray], kind: str, hidden_dim: int, input_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_rec = make_reservoir_matrix(kind, hidden_dim, seed=seed)
    w_in = rng.normal(0.0, 0.7 / math.sqrt(input_dim), size=(hidden_dim, input_dim)).astype(np.float32)
    states = []
    for x in xs:
        h = np.zeros(hidden_dim, dtype=np.float32)
        for frame in x:
            h = np.tanh(w_in @ frame + w_rec @ h)
        states.append(h)
    return np.asarray(states, dtype=np.float32), w_rec


def lin_sequence_matrix(sequence_r: int, sequence_l: int, hidden_dim: int) -> tuple[np.ndarray, int, int]:
    sequence_ell = sequence_r + sequence_l - 1
    band_width = max(1, hidden_dim // sequence_ell)
    active_dim = min(hidden_dim, band_width * sequence_ell)
    matrix = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    for band in range(sequence_ell - 1):
        source = slice(band * band_width, (band + 1) * band_width)
        target = slice((band + 1) * band_width, (band + 2) * band_width)
        matrix[target, source] = np.eye(band_width, dtype=np.float32)
    return matrix, band_width, active_dim


def lin_sequence_states(
    xs: list[np.ndarray],
    sequence_r: int,
    sequence_l: int,
    input_dim: int,
    seed: int,
    input_injection: str = "front_bands",
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w_rec, band_width, active_dim = lin_sequence_matrix(sequence_r, sequence_l, HIDDEN_DIM)
    w_in = np.zeros((HIDDEN_DIM, input_dim), dtype=np.float32)
    if input_injection == "front_bands":
        rows = min(active_dim, sequence_r * band_width)
        w_in[:rows] = rng.normal(0.0, 0.7 / math.sqrt(input_dim), size=(rows, input_dim))
    elif input_injection == "all_units":
        w_in[:active_dim] = rng.normal(0.0, 0.7 / math.sqrt(input_dim), size=(active_dim, input_dim))
    else:
        raise ValueError(f"unknown Lin input injection: {input_injection}")
    states = []
    for x in xs:
        h = np.zeros(HIDDEN_DIM, dtype=np.float32)
        for frame in x:
            h = np.tanh(w_in @ frame + w_rec @ h)
        states.append(h)
    return np.asarray(states, dtype=np.float32), w_rec


def lin_block_sequence_matrix(input_dim: int, sequence_r: int, sequence_l: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent finite chains, one per input feature."""
    sequence_ell = sequence_r + sequence_l - 1
    hidden_dim = input_dim * sequence_ell
    w_rec = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    w_in = np.zeros((hidden_dim, input_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(max(1, sequence_r))
    for feature_idx in range(input_dim):
        base = feature_idx * sequence_ell
        for step in range(sequence_ell - 1):
            w_rec[base + step + 1, base + step] = 1.0
        for step in range(sequence_r):
            w_in[base + step, feature_idx] = scale
    return w_rec, w_in


def lin_block_sequence_states(
    xs: list[np.ndarray],
    sequence_r: int,
    sequence_l: int,
) -> tuple[np.ndarray, np.ndarray]:
    w_rec, w_in = lin_block_sequence_matrix(xs[0].shape[1], sequence_r, sequence_l)
    states = []
    for x in xs:
        h = np.zeros(w_rec.shape[0], dtype=np.float32)
        for frame in x:
            h = np.tanh(w_in @ frame + w_rec @ h)
        states.append(h)
    return np.asarray(states, dtype=np.float32), w_rec


def matrix_metrics(matrix: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvals(matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    singular_sum = float(np.sum(singular))
    probs = singular / singular_sum if singular_sum > 1e-8 else np.ones_like(singular) / len(singular)
    effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    norm = float(np.linalg.norm(matrix)) + 1e-8
    powers = []
    current = np.eye(matrix.shape[0], dtype=np.float32)
    for _ in range(1, 16):
        current = current @ matrix
        powers.append(float(np.linalg.norm(current, ord=2)))
    return {
        "spectral_radius": float(np.max(np.abs(eigvals))) if eigvals.size else 0.0,
        "symmetry_index": float(np.linalg.norm(matrix - matrix.T) / norm),
        "normality_index": float(np.linalg.norm(matrix @ matrix.T - matrix.T @ matrix) / (norm * norm)),
        "effective_rank": effective_rank,
        "transient_amplification": max(powers) if powers else 0.0,
        "power_decay_10": powers[9] if len(powers) > 9 else float("nan"),
        "oscillatory_score": float(np.mean(np.abs(np.imag(eigvals)) > 0.05)) if eigvals.size else 0.0,
        "nilpotent_score": float(1.0 / (1.0 + (powers[9] if len(powers) > 9 else 0.0))),
    }


def greedy_success(dataset: Dataset, predict_fn: Callable[[list[np.ndarray]], np.ndarray], limit: int = ROLLOUT_LIMIT) -> float:
    graph = navigation_graph(dataset.task.env)
    state_index = {state: idx for idx, state in enumerate(sorted(dataset.task.env.states))}
    label_to_action = dataset.action_labels
    successes = 0
    total = min(limit, len(dataset.state_test))
    for idx in range(total):
        state = dataset.state_test[idx]
        goal = dataset.goal_test[idx]
        path = list(dataset.x_test[idx])
        max_steps = 12
        for _ in range(max_steps):
            if state == goal:
                successes += 1
                break
            pred = int(predict_fn([np.asarray(path, dtype=np.float32)])[0])
            action = label_to_action[pred] if pred < len(label_to_action) else label_to_action[0]
            next_candidates = [neighbor for edge_action, neighbor in graph[state] if edge_action == action]
            if not next_candidates:
                break
            state = next_candidates[0]
            frame = np.concatenate([
                state_feature(state, dataset.task, dataset.obs_mode, state_index),
                state_feature(goal, dataset.task, dataset.obs_mode, state_index),
            ])
            if dataset.temporal_mode != "temporal_full":
                state_dim = len(frame) // 2
                temporal_seed = stable_seed(
                    "rollout",
                    dataset.task.name,
                    dataset.obs_mode,
                    dataset.temporal_mode,
                    goal,
                    state,
                    len(path),
                )
                keep = temporal_keep_mask(len(path) + 1, dataset.temporal_mode, dataset.temporal_dropout_p, temporal_seed)[-1]
                if not keep:
                    frame[:state_dim] = 0.0
            path.append(frame)
        else:
            if state == goal:
                successes += 1
    return successes / total if total else float("nan")


def evaluate_dataset(dataset: Dataset, dataset_seed: int, seed: int = 0) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    x_train_last = pad_last_features(dataset.x_train)
    x_test_last = pad_last_features(dataset.x_test)
    y_train = dataset.y_train
    y_test = dataset.y_test

    memoryless_predictor = fit_linear_predictor(x_train_last, y_train)
    preds = memoryless_predictor(x_test_last)
    rows.append(result_row(dataset, "memoryless_linear", dataset_seed, seed, accuracy_score(y_test, list(preds)), greedy_success(dataset, lambda xs: memoryless_predictor(pad_last_features(xs)))))

    mlp_predictor = fit_mlp_predictor(x_train_last, y_train, seed=seed)
    mlp_preds = mlp_predictor(x_test_last)
    rows.append(result_row(dataset, "memoryless_mlp", dataset_seed, seed, accuracy_score(y_test, list(mlp_preds)), greedy_success(dataset, lambda xs: mlp_predictor(pad_last_features(xs)))))

    rnn = NumpyRNN(dataset.feature_dim, hidden_dim=HIDDEN_DIM, output_dim=len(dataset.action_labels), seed=seed)
    losses = rnn.fit(dataset.x_train, y_train, epochs=8, lr=0.015, seed=seed)
    rnn_preds = rnn.predict(dataset.x_test)
    rows.append(result_row(dataset, "numpy_rnn", dataset_seed, seed, accuracy_score(y_test, list(rnn_preds)), greedy_success(dataset, rnn.predict), train_loss=losses[-1]))
    metrics = matrix_metrics(rnn.w_rec)
    weight_rows.append({
        "task": dataset.task.name,
        "obs_mode": dataset.obs_mode,
        "temporal_mode": dataset.temporal_mode,
        "temporal_dropout_p": dataset.temporal_dropout_p,
        "model": "numpy_rnn",
        "sequence_r": "",
        "sequence_l": "",
        "sequence_ell": "",
        "input_injection": "",
        "dataset_seed": dataset_seed,
        "model_seed": seed,
        **metrics,
    })

    for kind in RESERVOIR_KINDS:
        for reservoir_seed in RESERVOIR_SEEDS:
            model_seed = seed + reservoir_seed
            train_states, w_rec = reservoir_states(dataset.x_train, kind, hidden_dim=HIDDEN_DIM, input_dim=dataset.feature_dim, seed=model_seed)
            test_states, _ = reservoir_states(dataset.x_test, kind, hidden_dim=HIDDEN_DIM, input_dim=dataset.feature_dim, seed=model_seed)
            reservoir_predictor = fit_linear_predictor(train_states, y_train)
            reservoir_preds = reservoir_predictor(test_states)

            def predict_reservoir(xs: list[np.ndarray], *, reservoir_kind: str = kind, fixed_seed: int = model_seed) -> np.ndarray:
                states, _ = reservoir_states(xs, reservoir_kind, hidden_dim=HIDDEN_DIM, input_dim=dataset.feature_dim, seed=fixed_seed)
                return reservoir_predictor(states)

            rows.append(result_row(dataset, f"reservoir_{kind}", dataset_seed, model_seed, accuracy_score(y_test, list(reservoir_preds)), greedy_success(dataset, predict_reservoir)))
            weight_rows.append({
                "task": dataset.task.name,
                "obs_mode": dataset.obs_mode,
                "temporal_mode": dataset.temporal_mode,
                "temporal_dropout_p": dataset.temporal_dropout_p,
                "model": f"reservoir_{kind}",
                "sequence_r": "",
                "sequence_l": "",
                "sequence_ell": "",
                "input_injection": "all_units",
                "dataset_seed": dataset_seed,
                "model_seed": model_seed,
                **matrix_metrics(w_rec),
            })

    lin_conditions = [(r, length, "front_bands") for r, length in LIN_SEQUENCE_CONFIGS]
    lin_conditions.append((*LIN_SEQUENCE_REFERENCE, "all_units"))
    for sequence_r, sequence_l, input_injection in lin_conditions:
        sequence_ell = sequence_r + sequence_l - 1
        model_name = f"lin_sequence_{input_injection}_r{sequence_r}_l{sequence_l}"
        for reservoir_seed in RESERVOIR_SEEDS:
            model_seed = seed + 500 + reservoir_seed + 17 * sequence_r + 31 * sequence_l
            train_states, w_rec = lin_sequence_states(
                dataset.x_train,
                sequence_r,
                sequence_l,
                dataset.feature_dim,
                model_seed,
                input_injection=input_injection,
            )
            test_states, _ = lin_sequence_states(
                dataset.x_test,
                sequence_r,
                sequence_l,
                dataset.feature_dim,
                model_seed,
                input_injection=input_injection,
            )
            sequence_predictor = fit_linear_predictor(train_states, y_train)
            sequence_preds = sequence_predictor(test_states)
            context_accuracy = float("nan")
            context_targets = behavioral_context_targets(dataset)
            if context_targets is not None:
                context_train, context_test = context_targets
                context_predictor = fit_linear_predictor(train_states, context_train)
                context_preds = context_predictor(test_states)
                context_accuracy = accuracy_score(context_test, list(context_preds))

            def predict_lin_sequence(
                xs: list[np.ndarray],
                *,
                fixed_r: int = sequence_r,
                fixed_l: int = sequence_l,
                fixed_seed: int = model_seed,
                fixed_injection: str = input_injection,
            ) -> np.ndarray:
                states, _ = lin_sequence_states(
                    xs,
                    fixed_r,
                    fixed_l,
                    dataset.feature_dim,
                    fixed_seed,
                    input_injection=fixed_injection,
                )
                return sequence_predictor(states)

            rows.append(result_row(
                dataset,
                model_name,
                dataset_seed,
                model_seed,
                accuracy_score(y_test, list(sequence_preds)),
                greedy_success(dataset, predict_lin_sequence),
                sequence_r=sequence_r,
                sequence_l=sequence_l,
                input_injection=input_injection,
                context_accuracy=context_accuracy,
            ))
            weight_rows.append({
                "task": dataset.task.name,
                "obs_mode": dataset.obs_mode,
                "temporal_mode": dataset.temporal_mode,
                "temporal_dropout_p": dataset.temporal_dropout_p,
                "model": model_name,
                "sequence_r": sequence_r,
                "sequence_l": sequence_l,
                "sequence_ell": sequence_ell,
                "input_injection": input_injection,
                "dataset_seed": dataset_seed,
                "model_seed": model_seed,
                **matrix_metrics(w_rec),
            })
    return rows, weight_rows


def result_row(
    dataset: Dataset,
    model: str,
    dataset_seed: int,
    model_seed: int,
    accuracy: float,
    success: float,
    train_loss: float = float("nan"),
    sequence_r: int | None = None,
    sequence_l: int | None = None,
    input_injection: str = "",
    context_accuracy: float = float("nan"),
) -> dict[str, object]:
    sequence_ell = sequence_r + sequence_l - 1 if sequence_r is not None and sequence_l is not None else ""
    return {
        "task": dataset.task.name,
        "geometry": dataset.task.geometry,
        "behavior_family": dataset.task.behavior_family,
        "obs_mode": dataset.obs_mode,
        "temporal_mode": dataset.temporal_mode,
        "temporal_dropout_p": dataset.temporal_dropout_p,
        "model": model,
        "sequence_r": sequence_r if sequence_r is not None else "",
        "sequence_l": sequence_l if sequence_l is not None else "",
        "sequence_ell": sequence_ell,
        "input_injection": input_injection,
        "dataset_seed": dataset_seed,
        "model_seed": model_seed,
        "n_train": len(dataset.y_train),
        "n_test": len(dataset.y_test),
        "n_actions": len(dataset.action_labels),
        "action_accuracy": accuracy,
        "greedy_success": success,
        "train_loss": train_loss,
        "context_accuracy": context_accuracy,
    }


def collect_results() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    all_weight_rows: list[dict[str, object]] = []
    for task_idx, task in enumerate(task_specs()):
        for obs_mode in OBS_MODES:
            for temporal_idx, temporal_mode in enumerate(TEMPORAL_MODES):
                for dataset_seed in DATASET_SEEDS:
                    dataset = build_dataset(
                        task,
                        obs_mode,
                        temporal_mode=temporal_mode,
                        temporal_dropout_p=TEMPORAL_DROPOUT_P,
                        seed=dataset_seed + task_idx,
                    )
                    rows, weight_rows = evaluate_dataset(
                        dataset,
                        dataset_seed=dataset_seed,
                        seed=100 + dataset_seed + task_idx + 1000 * temporal_idx,
                    )
                    all_rows.extend(rows)
                    all_weight_rows.extend(weight_rows)
    return all_rows, all_weight_rows


def evaluate_long_architecture_dataset(dataset: Dataset, dataset_seed: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_train_last = pad_last_features(dataset.x_train)
    x_test_last = pad_last_features(dataset.x_test)
    context_targets = behavioral_context_targets(dataset)
    memoryless_predictor = fit_linear_predictor(x_train_last, dataset.y_train)
    memoryless_preds = memoryless_predictor(x_test_last)
    rows.append(result_row(
        dataset,
        "memoryless_linear",
        dataset_seed,
        seed,
        accuracy_score(dataset.y_test, list(memoryless_preds)),
        float("nan"),
        context_accuracy=context_readout_accuracy(x_train_last, x_test_last, context_targets),
    ))

    mlp_predictor = fit_mlp_predictor(x_train_last, dataset.y_train, seed=seed)
    rows.append(result_row(
        dataset,
        "memoryless_mlp",
        dataset_seed,
        seed,
        accuracy_score(dataset.y_test, list(mlp_predictor(x_test_last))),
        float("nan"),
        context_accuracy=context_readout_accuracy(x_train_last, x_test_last, context_targets),
    ))

    for history_len in LONG_HISTORY_LENGTHS:
        train_history = pad_history_features(dataset.x_train, history_len)
        test_history = pad_history_features(dataset.x_test, history_len)
        history_predictor = fit_linear_predictor(train_history, dataset.y_train)
        rows.append(result_row(
            dataset,
            f"history_linear_h{history_len}",
            dataset_seed,
            seed,
            accuracy_score(dataset.y_test, list(history_predictor(test_history))),
            float("nan"),
            context_accuracy=context_readout_accuracy(train_history, test_history, context_targets),
        ))

    rnn = NumpyRNN(dataset.feature_dim, hidden_dim=HIDDEN_DIM, output_dim=len(dataset.action_labels), seed=seed)
    losses = rnn.fit(dataset.x_train, dataset.y_train, epochs=8, lr=0.015, seed=seed)
    train_rnn_states = rnn.final_states(dataset.x_train)
    test_rnn_states = rnn.final_states(dataset.x_test)
    rows.append(result_row(
        dataset,
        "numpy_rnn",
        dataset_seed,
        seed,
        accuracy_score(dataset.y_test, list(rnn.predict(dataset.x_test))),
        float("nan"),
        train_loss=losses[-1],
        context_accuracy=context_readout_accuracy(train_rnn_states, test_rnn_states, context_targets),
    ))

    for kind in RESERVOIR_KINDS:
        for reservoir_seed in RESERVOIR_SEEDS:
            model_seed = seed + reservoir_seed
            train_states, _ = reservoir_states(
                dataset.x_train,
                kind,
                hidden_dim=HIDDEN_DIM,
                input_dim=dataset.feature_dim,
                seed=model_seed,
            )
            test_states, _ = reservoir_states(
                dataset.x_test,
                kind,
                hidden_dim=HIDDEN_DIM,
                input_dim=dataset.feature_dim,
                seed=model_seed,
            )
            reservoir_predictor = fit_linear_predictor(train_states, dataset.y_train)
            rows.append(result_row(
                dataset,
                f"reservoir_{kind}",
                dataset_seed,
                model_seed,
                accuracy_score(dataset.y_test, list(reservoir_predictor(test_states))),
                float("nan"),
                input_injection="all_units",
                context_accuracy=context_readout_accuracy(train_states, test_states, context_targets),
            ))

    lin_conditions = [(r, length, "front_bands") for r, length in LIN_SEQUENCE_CONFIGS]
    lin_conditions.append((*LIN_SEQUENCE_REFERENCE, "all_units"))
    for sequence_r, sequence_l, input_injection in lin_conditions:
        for reservoir_seed in RESERVOIR_SEEDS:
            model_seed = seed + 500 + reservoir_seed + 17 * sequence_r + 31 * sequence_l
            train_states, _ = lin_sequence_states(
                dataset.x_train,
                sequence_r,
                sequence_l,
                dataset.feature_dim,
                model_seed,
                input_injection=input_injection,
            )
            test_states, _ = lin_sequence_states(
                dataset.x_test,
                sequence_r,
                sequence_l,
                dataset.feature_dim,
                model_seed,
                input_injection=input_injection,
            )
            sequence_predictor = fit_linear_predictor(train_states, dataset.y_train)
            rows.append(result_row(
                dataset,
                f"lin_sequence_{input_injection}_r{sequence_r}_l{sequence_l}",
                dataset_seed,
                model_seed,
                accuracy_score(dataset.y_test, list(sequence_predictor(test_states))),
                float("nan"),
                sequence_r=sequence_r,
                sequence_l=sequence_l,
                input_injection=input_injection,
                context_accuracy=context_readout_accuracy(train_states, test_states, context_targets),
            ))

    for sequence_r, sequence_l in LIN_LONG_SEQUENCE_CONFIGS:
        train_states, _ = lin_block_sequence_states(dataset.x_train, sequence_r, sequence_l)
        test_states, _ = lin_block_sequence_states(dataset.x_test, sequence_r, sequence_l)
        sequence_predictor = fit_linear_predictor(train_states, dataset.y_train)
        sequence_preds = sequence_predictor(test_states)
        rows.append(result_row(
            dataset,
            f"lin_block_r{sequence_r}_l{sequence_l}",
            dataset_seed,
            seed,
            accuracy_score(dataset.y_test, list(sequence_preds)),
            float("nan"),
            sequence_r=sequence_r,
            sequence_l=sequence_l,
            input_injection="independent_blocks",
            context_accuracy=context_readout_accuracy(train_states, test_states, context_targets),
        ))
    return rows


def collect_long_architecture_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_idx, task in enumerate(long_behavior_task_specs()):
        for obs_idx, obs_mode in enumerate(("sparse_aliased", "egocentric_landmark")):
            for temporal_idx, temporal_mode in enumerate(("temporal_full", "temporal_cue_then_blank")):
                for dataset_seed in LIN_LONG_DATASET_SEEDS:
                    dataset = build_dataset(
                        task,
                        obs_mode,
                        temporal_mode=temporal_mode,
                        temporal_dropout_p=TEMPORAL_DROPOUT_P,
                        seed=dataset_seed + task_idx,
                    )
                    rows.extend(evaluate_long_architecture_dataset(
                        dataset,
                        dataset_seed=dataset_seed,
                        seed=5000 + dataset_seed + 100 * obs_idx + 1000 * temporal_idx,
                    ))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["task"]),
            str(row["geometry"]),
            str(row["behavior_family"]),
            str(row["obs_mode"]),
            str(row["temporal_mode"]),
            str(row["model"]),
        )].append(row)
    summary: list[dict[str, object]] = []
    for (task, geometry, behavior_family, obs_mode, temporal_mode, model), group in sorted(grouped.items()):
        accuracies = np.asarray([float(row["action_accuracy"]) for row in group], dtype=float)
        successes = np.asarray([float(row["greedy_success"]) for row in group], dtype=float)
        success_values = successes[np.isfinite(successes)]
        contexts = np.asarray([float(row["context_accuracy"]) for row in group], dtype=float)
        context_values = contexts[np.isfinite(contexts)]
        first = group[0]
        summary.append({
            "task": task,
            "geometry": geometry,
            "behavior_family": behavior_family,
            "obs_mode": obs_mode,
            "temporal_mode": temporal_mode,
            "model": model,
            "sequence_r": first["sequence_r"],
            "sequence_l": first["sequence_l"],
            "sequence_ell": first["sequence_ell"],
            "input_injection": first["input_injection"],
            "n_runs": len(group),
            "action_accuracy_mean": float(np.mean(accuracies)),
            "action_accuracy_sem": float(np.std(accuracies, ddof=1) / math.sqrt(len(accuracies))) if len(accuracies) > 1 else 0.0,
            "greedy_success_mean": float(np.mean(success_values)) if len(success_values) else float("nan"),
            "greedy_success_sem": float(np.std(success_values, ddof=1) / math.sqrt(len(success_values))) if len(success_values) > 1 else 0.0,
            "context_accuracy_mean": float(np.mean(context_values)) if len(context_values) else float("nan"),
            "context_accuracy_sem": float(np.std(context_values, ddof=1) / math.sqrt(len(context_values))) if len(context_values) > 1 else 0.0,
        })
    return summary


def memory_advantage_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_task_obs: dict[tuple[str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summary:
        by_task_obs[(str(row["task"]), str(row["obs_mode"]), str(row["temporal_mode"]))][str(row["model"])] = row
    out: list[dict[str, object]] = []
    for (task, obs_mode, temporal_mode), model_rows in sorted(by_task_obs.items()):
        memoryless = max(
            float(model_rows[model]["action_accuracy_mean"])
            for model in ("memoryless_linear", "memoryless_mlp")
            if model in model_rows
        )
        recurrent_candidates = {
            model: row for model, row in model_rows.items()
            if model == "numpy_rnn" or model.startswith("reservoir_") or model.startswith("lin_sequence_")
        }
        best_model, best_row = max(
            recurrent_candidates.items(),
            key=lambda item: float(item[1]["action_accuracy_mean"]),
        )
        out.append({
            "task": task,
            "obs_mode": obs_mode,
            "temporal_mode": temporal_mode,
            "best_memoryless_accuracy": memoryless,
            "best_recurrent_model": best_model,
            "best_recurrent_accuracy": float(best_row["action_accuracy_mean"]),
            "memory_advantage": float(best_row["action_accuracy_mean"]) - memoryless,
        })
    return out


def heatmap(rows: list[dict[str, object]], *, obs_mode: str, temporal_mode: str, metric: str, path: Path) -> None:
    tasks = [task.name for task in task_specs()]
    models = ["memoryless_linear", "memoryless_mlp", "numpy_rnn", "reservoir_random", "reservoir_symmetric", "reservoir_oscillatory", "reservoir_nilpotent", "reservoir_block_hybrid"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["obs_mode"] == obs_mode and row["temporal_mode"] == temporal_mode:
            grouped[(str(row["task"]), str(row["model"]))].append(float(row[metric]))
    values = {key: float(np.mean(vals)) for key, vals in grouped.items()}
    width, height = 1320, 720
    margin_left, margin_right, margin_top, margin_bottom = 245, 50, 92, 230
    cell_w = (width - margin_left - margin_right) / len(tasks)
    cell_h = (height - margin_top - margin_bottom) / len(models)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (22, 163, 74)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:19px}.tick{font-size:17px;fill:#52606d}.cell{font-size:18px;font-weight:700}</style>',
        f'<text class="title" x="70" y="38">RNN/reservoir navigation benchmark: {esc(obs_mode)} / {esc(temporal_mode)}</text>',
        f'<text class="label" x="70" y="62">Cell = {esc(metric.replace("_", " "))}; supervised shortest-path imitation</text>',
    ]
    for row_idx, model in enumerate(models):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(model.replace("_", " "))}</text>')
        for col_idx, task in enumerate(tasks):
            value = values.get((task, model), float("nan"))
            x0 = margin_left + col_idx * cell_w
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text_color = "#ffffff" if value == value and value > 0.55 else "#1f2933"
            text = "NA" if value != value else f"{value:.2f}"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" fill="{text_color}">{text}</text>')
    for col_idx, task in enumerate(tasks):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="tick" text-anchor="end" x="{x0 + 26:.1f}" y="{height - 145}" transform="rotate(-32 {x0 + 26:.1f} {height - 145})">{esc(task.replace("_", " "))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def memory_advantage_plot(rows: list[dict[str, object]], path: Path) -> None:
    filtered = [
        row for row in rows
        if row["obs_mode"] in {"sparse_aliased", "hybrid"} and row["temporal_mode"] == "temporal_full"
    ]
    width, height = 1080, 760
    margin_left, margin_right, margin_top, margin_bottom = 210, 60, 78, 330
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_abs = max(0.05, max(abs(float(row["memory_advantage"])) for row in filtered))
    bar_w = plot_w / len(filtered)
    zero_y = margin_top + plot_h * 0.52
    scale = (plot_h * 0.45) / max_abs
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:18px}.tick{font-size:16px;fill:#52606d}.value{font-size:16px;font-weight:700}</style>',
        '<text class="title" x="70" y="38">Best recurrent memory advantage</text>',
        '<text class="label" x="70" y="62">Best recurrent action accuracy minus best memoryless action accuracy; two dataset seeds, three reservoir seeds</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" stroke="#9aa5b1" stroke-width="1"/>',
    ]
    for idx, row in enumerate(filtered):
        value = float(row["memory_advantage"])
        x = margin_left + idx * bar_w + bar_w * 0.16
        y = zero_y - max(0.0, value * scale)
        h = abs(value * scale)
        if value < 0:
            y = zero_y
        fill = "#15803d" if value >= 0 else "#b91c1c"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.68:.1f}" height="{max(1.0, h):.1f}" fill="{fill}" rx="2"/>')
        label_y = y - 5 if value >= 0 else y + h + 13
        parts.append(f'<text class="value" text-anchor="middle" x="{x + bar_w * 0.34:.1f}" y="{label_y:.1f}">{value:+.2f}</text>')
        task_label = f'{str(row["task"]).replace("_", " ")} / {str(row["obs_mode"]).replace("_", " ")}'
        tick_x = x + bar_w * 0.34
        parts.append(f'<text class="tick" text-anchor="end" x="{tick_x + 22:.1f}" y="{height - 180}" transform="rotate(-36 {tick_x + 22:.1f} {height - 180})">{esc(task_label)}</text>')
    parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 12}" y="{zero_y + 4:.1f}">0</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def temporal_sparsity_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        grouped[(str(row["obs_mode"]), str(row["temporal_mode"]), str(row["model"]))].append(row)
    out: list[dict[str, object]] = []
    for (obs_mode, temporal_mode, model), group in sorted(grouped.items()):
        accuracies = np.asarray([float(row["action_accuracy_mean"]) for row in group], dtype=float)
        successes = np.asarray([float(row["greedy_success_mean"]) for row in group], dtype=float)
        out.append({
            "obs_mode": obs_mode,
            "temporal_mode": temporal_mode,
            "model": model,
            "n_tasks": len(group),
            "action_accuracy_mean": float(np.mean(accuracies)),
            "action_accuracy_sem": float(np.std(accuracies, ddof=1) / math.sqrt(len(accuracies))) if len(accuracies) > 1 else 0.0,
            "greedy_success_mean": float(np.mean(successes)),
            "greedy_success_sem": float(np.std(successes, ddof=1) / math.sqrt(len(successes))) if len(successes) > 1 else 0.0,
        })
    return out


def temporal_sparsity_advantage(advantage_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in advantage_rows:
        grouped[(str(row["obs_mode"]), str(row["temporal_mode"]))].append(row)
    out: list[dict[str, object]] = []
    for (obs_mode, temporal_mode), group in sorted(grouped.items()):
        advantages = np.asarray([float(row["memory_advantage"]) for row in group], dtype=float)
        best = max(group, key=lambda row: float(row["memory_advantage"]))
        out.append({
            "obs_mode": obs_mode,
            "temporal_mode": temporal_mode,
            "n_tasks": len(group),
            "memory_advantage_mean": float(np.mean(advantages)),
            "memory_advantage_sem": float(np.std(advantages, ddof=1) / math.sqrt(len(advantages))) if len(advantages) > 1 else 0.0,
            "positive_task_count": int(np.sum(advantages > 0.02)),
            "best_task": best["task"],
            "best_recurrent_model": best["best_recurrent_model"],
            "best_task_advantage": float(best["memory_advantage"]),
        })
    return out


def temporal_reservoir_plot(summary: list[dict[str, object]], path: Path, obs_mode: str = "sparse_aliased") -> None:
    reservoirs = [
        "reservoir_random",
        "reservoir_symmetric",
        "reservoir_oscillatory",
        "reservoir_orthogonal",
        "reservoir_nilpotent",
        "reservoir_diagonal",
        "reservoir_block_hybrid",
    ]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary:
        if row["obs_mode"] == obs_mode and row["model"] in reservoirs:
            grouped[(str(row["model"]), str(row["temporal_mode"]))].append(float(row["action_accuracy_mean"]))
    values = {key: float(np.mean(vals)) for key, vals in grouped.items()}
    width, height = 760, 470
    margin_left, margin_right, margin_top, margin_bottom = 195, 50, 84, 78
    cell_w = (width - margin_left - margin_right) / len(TEMPORAL_MODES)
    cell_h = (height - margin_top - margin_bottom) / len(reservoirs)

    def color(value: float) -> str:
        low = (248, 250, 252)
        high = (37, 99, 235)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:31px;font-weight:700}.label{font-size:18px}.cell{font-size:18px;font-weight:700}</style>',
        '<text class="title" x="56" y="38">Temporal sparsity x reservoir type</text>',
        f'<text class="label" x="56" y="62">Mean action accuracy across tasks; observation mode = {esc(obs_mode)}</text>',
    ]
    for row_idx, model in enumerate(reservoirs):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 12}" y="{y0 + cell_h / 2 + 5:.1f}">{esc(model.replace("reservoir_", "").replace("_", " "))}</text>')
        for col_idx, temporal_mode in enumerate(TEMPORAL_MODES):
            value = values.get((model, temporal_mode), float("nan"))
            x0 = margin_left + col_idx * cell_w
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text = "NA" if value != value else f"{value:.2f}"
            text_color = "#ffffff" if value == value and value > 0.62 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" fill="{text_color}">{text}</text>')
    for col_idx, temporal_mode in enumerate(TEMPORAL_MODES):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="label" text-anchor="middle" x="{x0:.1f}" y="{height - 35}">{esc(temporal_mode.replace("temporal_", "").replace("_", " "))}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def lin_sequence_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in summary if str(row["model"]).startswith("lin_sequence_")]


def lin_sequence_injection_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in lin_sequence_summary(summary):
        if int(row["sequence_r"]) == LIN_SEQUENCE_REFERENCE[0] and int(row["sequence_l"]) == LIN_SEQUENCE_REFERENCE[1]:
            grouped[(str(row["obs_mode"]), str(row["temporal_mode"]), str(row["input_injection"]))].append(row)
    out: list[dict[str, object]] = []
    for (obs_mode, temporal_mode, input_injection), group in sorted(grouped.items()):
        accuracies = np.asarray([float(row["action_accuracy_mean"]) for row in group], dtype=float)
        contexts = np.asarray([float(row["context_accuracy_mean"]) for row in group], dtype=float)
        context_values = contexts[np.isfinite(contexts)]
        out.append({
            "obs_mode": obs_mode,
            "temporal_mode": temporal_mode,
            "sequence_r": LIN_SEQUENCE_REFERENCE[0],
            "sequence_l": LIN_SEQUENCE_REFERENCE[1],
            "sequence_ell": sum(LIN_SEQUENCE_REFERENCE) - 1,
            "input_injection": input_injection,
            "n_tasks": len(group),
            "action_accuracy_mean": float(np.mean(accuracies)),
            "action_accuracy_sem": float(np.std(accuracies, ddof=1) / math.sqrt(len(accuracies))) if len(accuracies) > 1 else 0.0,
            "behavioral_context_accuracy_mean": float(np.mean(context_values)) if len(context_values) else float("nan"),
        })
    return out


def lin_sequence_grid_plot(summary: list[dict[str, object]], path: Path) -> None:
    filtered = [
        row for row in lin_sequence_summary(summary)
        if row["input_injection"] == "front_bands"
        and row["obs_mode"] == "sparse_aliased"
        and row["temporal_mode"] == "temporal_cue_then_blank"
        and row["behavior_family"] not in {"formal", "metric_landmark"}
    ]
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in filtered:
        grouped[(int(row["sequence_r"]), int(row["sequence_l"]))].append(float(row["action_accuracy_mean"]))
    values = {key: float(np.mean(vals)) for key, vals in grouped.items()}
    rs = (1, 3, 5)
    lengths = (2, 4, 8)
    width, height = 580, 430
    margin_left, margin_right, margin_top, margin_bottom = 150, 56, 92, 80
    cell_w = (width - margin_left - margin_right) / len(lengths)
    cell_h = (height - margin_top - margin_bottom) / len(rs)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (21, 128, 61)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:30px;font-weight:700}.label{font-size:19px}.cell{font-size:20px;font-weight:700}</style>',
        '<text class="title" x="50" y="36">Lin-style sequence grid</text>',
        '<text class="label" x="50" y="60">Sparse/aliased cue-then-blank behavioral tasks; cell = action accuracy</text>',
    ]
    for row_idx, sequence_r in enumerate(rs):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">R = {sequence_r}</text>')
        for col_idx, sequence_l in enumerate(lengths):
            value = values.get((sequence_r, sequence_l), float("nan"))
            x0 = margin_left + col_idx * cell_w
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text = "NA" if value != value else f"{value:.2f}"
            text_color = "#ffffff" if value == value and value > 0.58 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 6:.1f}" fill="{text_color}">{text}</text>')
    for col_idx, sequence_l in enumerate(lengths):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="label" text-anchor="middle" x="{x0:.1f}" y="{height - 38}">L = {sequence_l}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def lin_long_sequence_plot(summary: list[dict[str, object]], path: Path) -> None:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in summary:
        if (
            str(row["model"]).startswith("lin_block_")
            and row["obs_mode"] == "sparse_aliased"
            and row["temporal_mode"] == "temporal_cue_then_blank"
        ):
            grouped[(int(row["sequence_r"]), int(row["sequence_l"]))].append(float(row["action_accuracy_mean"]))
    values = {key: float(np.mean(vals)) for key, vals in grouped.items()}
    rs = (1, 3, 5)
    lengths = (4, 16, 64)
    width, height = 620, 430
    margin_left, margin_right, margin_top, margin_bottom = 150, 56, 92, 80
    cell_w = (width - margin_left - margin_right) / len(lengths)
    cell_h = (height - margin_top - margin_bottom) / len(rs)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (22, 101, 52)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:30px;font-weight:700}.label{font-size:19px}.cell{font-size:20px;font-weight:700}</style>',
        '<text class="title" x="50" y="36">Long independent Lin blocks</text>',
        '<text class="label" x="50" y="60">72-step delayed cue task; sparse cue-then-blank; cell = action accuracy</text>',
    ]
    for row_idx, sequence_r in enumerate(rs):
        y0 = margin_top + row_idx * cell_h
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 5:.1f}">R = {sequence_r}</text>')
        for col_idx, sequence_l in enumerate(lengths):
            value = values.get((sequence_r, sequence_l), float("nan"))
            x0 = margin_left + col_idx * cell_w
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text = "NA" if value != value else f"{value:.2f}"
            text_color = "#ffffff" if value == value and value > 0.58 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 6:.1f}" fill="{text_color}">{text}</text>')
    for col_idx, sequence_l in enumerate(lengths):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        parts.append(f'<text class="label" text-anchor="middle" x="{x0:.1f}" y="{height - 38}">L = {sequence_l}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def long_architecture_family(model: str) -> str:
    if model.startswith("history_linear_"):
        return "explicit_observed_history"
    if model.startswith("reservoir_"):
        return model
    if model.startswith("lin_sequence_"):
        return "shared_lin_sequence"
    if model.startswith("lin_block_"):
        return "independent_lin_block"
    return model


def long_architecture_family_summary(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        grouped[(
            str(row["task"]),
            str(row["obs_mode"]),
            str(row["temporal_mode"]),
            long_architecture_family(str(row["model"])),
        )].append(row)
    out: list[dict[str, object]] = []
    for (task, obs_mode, temporal_mode, family), group in sorted(grouped.items()):
        best = max(group, key=lambda row: float(row["action_accuracy_mean"]))
        out.append({
            "task": task,
            "obs_mode": obs_mode,
            "temporal_mode": temporal_mode,
            "architecture_family": family,
            "n_models": len(group),
            "best_model": best["model"],
            "best_sequence_r": best["sequence_r"],
            "best_sequence_l": best["sequence_l"],
            "best_action_accuracy_mean": best["action_accuracy_mean"],
            "best_action_accuracy_sem": best["action_accuracy_sem"],
            "best_context_accuracy_mean": best["context_accuracy_mean"],
        })
    return out


def long_architecture_plot(summary: list[dict[str, object]], path: Path) -> None:
    regimes = [
        ("sparse_aliased", "temporal_full"),
        ("sparse_aliased", "temporal_cue_then_blank"),
        ("egocentric_landmark", "temporal_full"),
        ("egocentric_landmark", "temporal_cue_then_blank"),
    ]
    models = [
        "memoryless_linear",
        "memoryless_mlp",
        *(f"history_linear_h{history_len}" for history_len in LONG_HISTORY_LENGTHS),
        "numpy_rnn",
        *(f"reservoir_{kind}" for kind in RESERVOIR_KINDS),
        *(f"lin_sequence_front_bands_r{sequence_r}_l{sequence_l}" for sequence_r, sequence_l in LIN_SEQUENCE_CONFIGS),
        f"lin_sequence_all_units_r{LIN_SEQUENCE_REFERENCE[0]}_l{LIN_SEQUENCE_REFERENCE[1]}",
        *(f"lin_block_r{sequence_r}_l{sequence_l}" for sequence_r, sequence_l in LIN_LONG_SEQUENCE_CONFIGS),
    ]
    values = {
        (str(row["model"]), str(row["obs_mode"]), str(row["temporal_mode"])): float(row["action_accuracy_mean"])
        for row in summary
    }
    width, height = 1080, 1650
    margin_left, margin_right, margin_top, margin_bottom = 390, 54, 108, 350
    cell_w = (width - margin_left - margin_right) / len(regimes)
    cell_h = (height - margin_top - margin_bottom) / len(models)

    def color(value: float) -> str:
        low = (254, 242, 242)
        high = (22, 101, 52)
        r = round(low[0] + (high[0] - low[0]) * value)
        g = round(low[1] + (high[1] - low[1]) * value)
        b = round(low[2] + (high[2] - low[2]) * value)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}.title{font-size:32px;font-weight:700}.label{font-size:18px}.cell{font-size:17px;font-weight:700}</style>',
        '<text class="title" x="54" y="38">Long delayed navigation architecture comparison</text>',
        '<text class="label" x="54" y="64">72-step delayed T-maze; cell = mean supervised action accuracy</text>',
    ]
    for row_idx, model in enumerate(models):
        y0 = margin_top + row_idx * cell_h
        label = model.replace("lin_sequence_", "shared_lin_").replace("lin_block_", "independent_lin_").replace("_", " ")
        parts.append(f'<text class="label" text-anchor="end" x="{margin_left - 14}" y="{y0 + cell_h / 2 + 4:.1f}">{esc(label)}</text>')
        for col_idx, (obs_mode, temporal_mode) in enumerate(regimes):
            x0 = margin_left + col_idx * cell_w
            value = values.get((model, obs_mode, temporal_mode), float("nan"))
            fill = "#f5f7fa" if value != value else color(value)
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')
            text = "NA" if value != value else f"{value:.2f}"
            text_color = "#ffffff" if value == value and value > 0.60 else "#1f2933"
            parts.append(f'<text class="cell" text-anchor="middle" x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 4:.1f}" fill="{text_color}">{text}</text>')
    for col_idx, (obs_mode, temporal_mode) in enumerate(regimes):
        x0 = margin_left + col_idx * cell_w + cell_w / 2
        label = f'{obs_mode.replace("_", " ")} / {temporal_mode.replace("temporal_", "").replace("_", " ")}'
        parts.append(f'<text class="label" text-anchor="end" x="{x0 + 42:.1f}" y="{height - 268}" transform="rotate(-34 {x0 + 42:.1f} {height - 268})">{esc(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows, weight_rows = collect_results()
    summary_rows = summarize_rows(rows)
    advantage_rows = memory_advantage_summary(summary_rows)
    temporal_summary_rows = temporal_sparsity_summary(summary_rows)
    temporal_advantage_rows = temporal_sparsity_advantage(advantage_rows)
    lin_rows = lin_sequence_summary(summary_rows)
    lin_injection_rows = lin_sequence_injection_summary(summary_rows)
    long_lin_rows = collect_long_architecture_results()
    long_lin_summary_rows = summarize_rows(long_lin_rows)
    long_architecture_family_rows = long_architecture_family_summary(long_lin_summary_rows)
    write_csv(FIGURES / "rnn_navigation_benchmark.csv", rows)
    write_csv(FIGURES / "rnn_navigation_summary.csv", summary_rows)
    write_csv(FIGURES / "rnn_memory_advantage.csv", advantage_rows)
    write_csv(FIGURES / "rnn_temporal_sparsity_summary.csv", temporal_summary_rows)
    write_csv(FIGURES / "rnn_temporal_sparsity_advantage.csv", temporal_advantage_rows)
    write_csv(FIGURES / "rnn_lin_sequence_summary.csv", lin_rows)
    write_csv(FIGURES / "rnn_lin_sequence_injection.csv", lin_injection_rows)
    write_csv(FIGURES / "rnn_lin_long_sequence_benchmark.csv", long_lin_rows)
    write_csv(FIGURES / "rnn_lin_long_sequence_summary.csv", long_lin_summary_rows)
    write_csv(FIGURES / "rnn_lin_long_architecture_families.csv", long_architecture_family_rows)
    write_csv(FIGURES / "rnn_weight_analysis.csv", weight_rows)
    for obs_mode in OBS_MODES:
        heatmap(rows, obs_mode=obs_mode, temporal_mode="temporal_full", metric="action_accuracy", path=FIGURES / f"rnn_navigation_{obs_mode}_accuracy.svg")
        heatmap(rows, obs_mode=obs_mode, temporal_mode="temporal_full", metric="greedy_success", path=FIGURES / f"rnn_navigation_{obs_mode}_success.svg")
    memory_advantage_plot(advantage_rows, FIGURES / "rnn_memory_advantage.svg")
    temporal_reservoir_plot(summary_rows, FIGURES / "rnn_temporal_sparsity_reservoirs.svg")
    lin_sequence_grid_plot(summary_rows, FIGURES / "rnn_lin_sequence_grid.svg")
    lin_long_sequence_plot(long_lin_summary_rows, FIGURES / "rnn_lin_long_sequence_grid.svg")
    long_architecture_plot(long_lin_summary_rows, FIGURES / "rnn_lin_long_architecture_comparison.svg")
    print("Generated:")
    print((FIGURES / "rnn_navigation_benchmark.csv").relative_to(ROOT))
    print((FIGURES / "rnn_navigation_summary.csv").relative_to(ROOT))
    print((FIGURES / "rnn_memory_advantage.csv").relative_to(ROOT))
    print((FIGURES / "rnn_temporal_sparsity_summary.csv").relative_to(ROOT))
    print((FIGURES / "rnn_temporal_sparsity_advantage.csv").relative_to(ROOT))
    print((FIGURES / "rnn_lin_sequence_summary.csv").relative_to(ROOT))
    print((FIGURES / "rnn_lin_sequence_injection.csv").relative_to(ROOT))
    print((FIGURES / "rnn_lin_long_sequence_benchmark.csv").relative_to(ROOT))
    print((FIGURES / "rnn_lin_long_sequence_summary.csv").relative_to(ROOT))
    print((FIGURES / "rnn_lin_long_architecture_families.csv").relative_to(ROOT))
    print((FIGURES / "rnn_weight_analysis.csv").relative_to(ROOT))
    for obs_mode in OBS_MODES:
        print((FIGURES / f"rnn_navigation_{obs_mode}_accuracy.svg").relative_to(ROOT))
        print((FIGURES / f"rnn_navigation_{obs_mode}_success.svg").relative_to(ROOT))
    print((FIGURES / "rnn_memory_advantage.svg").relative_to(ROOT))
    print((FIGURES / "rnn_temporal_sparsity_reservoirs.svg").relative_to(ROOT))
    print((FIGURES / "rnn_lin_sequence_grid.svg").relative_to(ROOT))
    print((FIGURES / "rnn_lin_long_sequence_grid.svg").relative_to(ROOT))
    print((FIGURES / "rnn_lin_long_architecture_comparison.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
