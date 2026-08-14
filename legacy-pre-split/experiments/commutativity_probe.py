"""Probe effective commutativity in simple navigation graphs.

This scaffold operationalizes one GeoMem theory variable: whether action order
can be compressed away. It reports both raw scores and a valid-transition score
that filters out trajectories where either action order hits a boundary or
otherwise invalid transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Callable


State = tuple[int, ...]
Action = str
Transition = dict[tuple[State, Action], State]


@dataclass(frozen=True)
class Environment:
    name: str
    states: tuple[State, ...]
    actions: tuple[Action, ...]
    transition: Transition
    valid: frozenset[tuple[State, Action]]
    max_depth: int | None = None

    def is_valid(self, state: State, action: Action) -> bool:
        return (state, action) in self.valid

    def step(self, state: State, action: Action) -> State:
        return self.transition.get((state, action), state)

    def rollout(self, start: State, actions: tuple[Action, ...]) -> tuple[State, bool]:
        state = start
        valid_path = True
        for action in actions:
            valid_path = valid_path and self.is_valid(state, action)
            state = self.step(state, action)
        return state, valid_path


def _valid_from_nonself(transition: Transition) -> frozenset[tuple[State, Action]]:
    return frozenset(key for key, target in transition.items() if key[0] != target)


def torus_lattice(width: int = 5, height: int = 5) -> Environment:
    """A periodic 2D lattice; cardinal moves commute exactly."""
    actions = ("N", "S", "E", "W")
    states = tuple((x, y) for x in range(width) for y in range(height))
    transition: Transition = {}
    for x, y in states:
        transition[((x, y), "N")] = (x, (y + 1) % height)
        transition[((x, y), "S")] = (x, (y - 1) % height)
        transition[((x, y), "E")] = ((x + 1) % width, y)
        transition[((x, y), "W")] = ((x - 1) % width, y)
    return Environment("torus_lattice", states, actions, transition, frozenset(transition))


def clamped_lattice(width: int = 5, height: int = 5) -> Environment:
    """A bounded 2D lattice; boundary clamping breaks ideal commutativity."""
    actions = ("N", "S", "E", "W")
    states = tuple((x, y) for x in range(width) for y in range(height))
    transition: Transition = {}
    for x, y in states:
        transition[((x, y), "N")] = (x, min(height - 1, y + 1))
        transition[((x, y), "S")] = (x, max(0, y - 1))
        transition[((x, y), "E")] = (min(width - 1, x + 1), y)
        transition[((x, y), "W")] = (max(0, x - 1), y)
    return Environment("clamped_lattice", states, actions, transition, _valid_from_nonself(transition))


def lattice(width: int = 5, height: int = 5) -> Environment:
    """Backward-compatible alias for the old bounded lattice."""
    return clamped_lattice(width=width, height=height)


def tree(branching: int = 3, depth: int = 4) -> Environment:
    actions = tuple(f"C{i}" for i in range(branching)) + ("U",)
    states: list[State] = [()]
    frontier: list[State] = [()]
    for _ in range(depth):
        next_frontier: list[State] = []
        for state in frontier:
            for i in range(branching):
                child = state + (i,)
                states.append(child)
                next_frontier.append(child)
        frontier = next_frontier

    state_set = set(states)
    transition: Transition = {}
    for state in states:
        for i in range(branching):
            child = state + (i,)
            transition[(state, f"C{i}")] = child if child in state_set else state
        transition[(state, "U")] = state[:-1] if state else state
    return Environment("tree", tuple(states), actions, transition, _valid_from_nonself(transition), depth)


def shortcut_tree(branching: int = 3, depth: int = 4, shortcut_density: float = 1.0) -> Environment:
    """A tree with deterministic lateral shortcuts between nodes at the same depth.

    The base tree parameters match ``tree()`` by default. ``shortcut_density``
    controls the fraction of adjacent same-depth ring edges that are active.
    """
    if not 0.0 <= shortcut_density <= 1.0:
        raise ValueError("shortcut_density must be in [0, 1]")

    base = tree(branching=branching, depth=depth)
    actions = base.actions + ("L", "R")
    states = base.states
    transition = dict(base.transition)
    valid = set(base.valid)
    by_depth: dict[int, list[State]] = {}
    for state in states:
        by_depth.setdefault(len(state), []).append(state)

    for level_states in by_depth.values():
        level_states.sort()
        n = len(level_states)
        if n <= 1:
            for state in level_states:
                transition[(state, "L")] = state
                transition[(state, "R")] = state
            continue
        active_edges = round(shortcut_density * n)
        for idx, state in enumerate(level_states):
            left = level_states[(idx - 1) % n]
            right = level_states[(idx + 1) % n]
            if idx < active_edges:
                transition[(state, "R")] = right
                valid.add((state, "R"))
            else:
                transition[(state, "R")] = state
            if (idx - 1) % n < active_edges:
                transition[(state, "L")] = left
                valid.add((state, "L"))
            else:
                transition[(state, "L")] = state
    suffix = int(round(shortcut_density * 100))
    return Environment(f"shortcut_tree_{suffix:03d}", states, actions, transition, frozenset(valid), depth)


def bottleneck_rooms(size: int = 3) -> Environment:
    """Two local square rooms connected by a single door edge."""
    actions = ("N", "S", "E", "W")
    left = tuple((0, x, y) for x in range(size) for y in range(size))
    right = tuple((1, x, y) for x in range(size) for y in range(size))
    states = left + right
    state_set = set(states)
    transition: Transition = {}
    for room, x, y in states:
        candidates = {
            "N": (room, x, min(size - 1, y + 1)),
            "S": (room, x, max(0, y - 1)),
            "E": (room, min(size - 1, x + 1), y),
            "W": (room, max(0, x - 1), y),
        }
        for action, target in candidates.items():
            transition[((room, x, y), action)] = target if target in state_set else (room, x, y)

    door_left = (0, size - 1, size // 2)
    door_right = (1, 0, size // 2)
    transition[(door_left, "E")] = door_right
    transition[(door_right, "W")] = door_left
    return Environment("bottleneck_rooms", states, actions, transition, _valid_from_nonself(transition))


def trace_monoid(commuting_pairs: frozenset[tuple[Action, Action]], max_depth: int = 8) -> Environment:
    """Action-history environment quotienting by selected commutation relations.

    States are canonical action strings. If actions A and B commute, adjacent
    ``BA`` is rewritten to ``AB`` according to the action order. With no
    commuting pairs the state space is an ordered action tree. With all pairs
    commuting the state is equivalent to action counts.
    """
    actions = ("A", "B", "C")
    action_rank = {action: idx for idx, action in enumerate(actions)}
    symmetric_pairs = frozenset(
        tuple(sorted(pair, key=action_rank.__getitem__)) for pair in commuting_pairs
    )

    def commutes(first: Action, second: Action) -> bool:
        if first == second:
            return True
        return tuple(sorted((first, second), key=action_rank.__getitem__)) in symmetric_pairs

    def canonicalize(word: tuple[Action, ...]) -> State:
        ordered = list(word)
        changed = True
        while changed:
            changed = False
            for idx in range(len(ordered) - 1):
                left = ordered[idx]
                right = ordered[idx + 1]
                if action_rank[left] > action_rank[right] and commutes(left, right):
                    ordered[idx], ordered[idx + 1] = right, left
                    changed = True
        return tuple(action_rank[action] for action in ordered)

    states_set: set[State] = {()}
    frontier: set[State] = {()}
    for _ in range(max_depth):
        next_frontier: set[State] = set()
        for state in frontier:
            word = tuple(actions[idx] for idx in state)
            for action in actions:
                target = canonicalize(word + (action,))
                next_frontier.add(target)
                states_set.add(target)
        frontier = next_frontier

    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
    for state in states_set:
        word = tuple(actions[idx] for idx in state)
        for action in actions:
            if len(state) >= max_depth:
                transition[(state, action)] = state
            else:
                target = canonicalize(word + (action,))
                transition[(state, action)] = target
                valid.add((state, action))

    total_pairs = len(actions) * (len(actions) - 1) // 2
    suffix = int(round(100 * len(symmetric_pairs) / total_pairs))
    return Environment(f"trace_commute_{suffix:03d}", tuple(sorted(states_set)), actions, transition, frozenset(valid), max_depth)


def trace_monoid_family(max_depth: int = 8) -> list[Environment]:
    return [
        trace_monoid(frozenset(), max_depth=max_depth),
        trace_monoid(frozenset({("A", "B")}), max_depth=max_depth),
        trace_monoid(frozenset({("A", "B"), ("A", "C")}), max_depth=max_depth),
        trace_monoid(frozenset({("A", "B"), ("A", "C"), ("B", "C")}), max_depth=max_depth),
    ]


def action_strings(actions: tuple[Action, ...], length: int, rng: Random, n: int) -> list[tuple[Action, ...]]:
    return [tuple(rng.choice(actions) for _ in range(length)) for _ in range(n)]


def swapped_pairs(seq: tuple[Action, ...]) -> list[tuple[tuple[Action, ...], tuple[Action, ...]]]:
    pairs = []
    for i, j in combinations(range(len(seq)), 2):
        if seq[i] == seq[j]:
            continue
        swapped = list(seq)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        pairs.append((seq, tuple(swapped)))
    return pairs


def effective_commutativity(
    env: Environment,
    *,
    length: int = 4,
    n_sequences: int = 200,
    seed: int = 1,
    valid_only: bool = False,
    equivalent: Callable[[State, State], bool] | None = None,
) -> float:
    """Return fraction of action-order swaps that preserve final state.

    If ``valid_only`` is true, pairs are counted only when both action orders
    avoid invalid transitions such as clamped boundary moves.
    """
    rng = Random(seed)
    equivalent = equivalent or (lambda a, b: a == b)
    total = 0
    preserved = 0
    starts = list(env.states)
    if valid_only and env.max_depth is not None:
        starts = [state for state in starts if len(state) + length <= env.max_depth]
        if not starts:
            return float("nan")
    for seq in action_strings(env.actions, length, rng, n_sequences):
        start = rng.choice(starts)
        for original, swapped in swapped_pairs(seq):
            a, valid_a = env.rollout(start, original)
            b, valid_b = env.rollout(start, swapped)
            if valid_only and not (valid_a and valid_b):
                continue
            total += 1
            preserved += int(equivalent(a, b))
    return preserved / total if total else float("nan")


def pairwise_commutator_matrix(env: Environment, *, valid_only: bool = False) -> dict[tuple[Action, Action], float]:
    """Compute Pr[AB(s) == BA(s)] for every action pair over all states."""
    matrix: dict[tuple[Action, Action], float] = {}
    for first in env.actions:
        for second in env.actions:
            total = 0
            preserved = 0
            for state in env.states:
                ab, valid_ab = env.rollout(state, (first, second))
                ba, valid_ba = env.rollout(state, (second, first))
                if valid_only and not (valid_ab and valid_ba):
                    continue
                total += 1
                preserved += int(ab == ba)
            matrix[(first, second)] = preserved / total if total else float("nan")
    return matrix


def default_environments() -> list[Environment]:
    return [
        torus_lattice(),
        clamped_lattice(),
        bottleneck_rooms(),
        tree(branching=3, depth=4),
        shortcut_tree(branching=3, depth=4, shortcut_density=0.25),
        shortcut_tree(branching=3, depth=4, shortcut_density=0.50),
        shortcut_tree(branching=3, depth=4, shortcut_density=1.00),
        *trace_monoid_family(max_depth=8),
    ]


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.3f}"


def main() -> None:
    print("environment,raw_commutativity,valid_only_commutativity")
    for env in default_environments():
        raw = effective_commutativity(env)
        valid = effective_commutativity(env, valid_only=True)
        print(f"{env.name},{_fmt(raw)},{_fmt(valid)}")


if __name__ == "__main__":
    main()
