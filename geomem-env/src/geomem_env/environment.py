"""Probe effective commutativity in simple navigation graphs.

This scaffold operationalizes one GeoMem theory variable: whether action order
can be compressed away. It reports both raw scores and a valid-transition score
that filters out trajectories where either action order hits a boundary or
otherwise invalid transition.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from math import log2
from random import Random
from typing import Callable, Hashable


State = tuple[int, ...]
Action = str
Transition = dict[tuple[State, Action], State]
Observation = Hashable
ObservationFn = Callable[[State], Observation]
Policy = frozenset[str]


@dataclass(frozen=True)
class EnvironmentMetadata:
    family: str = "unspecified"
    geometry: str = "unspecified"
    observation: str = "latent_state"
    memory_pressure: str = "unspecified"
    axes: tuple[tuple[str, object], ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    terminated: bool
    truncated: bool
    valid: bool
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    family: str
    constructor: Callable[[], "Environment"]
    scale: str = "diagnostic"
    description: str = ""


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    observe: ObservationFn
    description: str = ""


@dataclass(frozen=True)
class ObservationChannel:
    name: str
    kind: str
    dim: int
    observe: Callable[[State, int | None], tuple[float, ...]]
    metadata: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class Environment:
    name: str
    states: tuple[State, ...]
    actions: tuple[Action, ...]
    transition: Transition
    valid: frozenset[tuple[State, Action]]
    max_depth: int | None = None
    metadata: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    start_states: tuple[State, ...] = ()
    goal_states: tuple[State, ...] = ()

    def is_valid(self, state: State, action: Action) -> bool:
        return (state, action) in self.valid

    def valid_actions(self, state: State) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if self.is_valid(state, action))

    def step(self, state: State, action: Action) -> State:
        return self.transition.get((state, action), state)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
        start: State | None = None,
    ) -> tuple[State, dict[str, object]]:
        """Stateless reset helper shaped like Gymnasium's reset return."""
        if options and "start" in options:
            candidate = options["start"]
            if not isinstance(candidate, tuple):
                raise TypeError("reset option 'start' must be a state tuple")
            start = candidate
        state = start if start is not None else (self.start_states[0] if self.start_states else self.states[0])
        if state not in self.states:
            raise ValueError(f"unknown start state for {self.name}: {state}")
        return state, {"seed": seed, "valid_actions": self.valid_actions(state)}

    def step_result(self, state: State, action: Action) -> StepResult:
        """Stateless transition helper carrying Gymnasium-style fields."""
        valid = self.is_valid(state, action)
        next_state = self.step(state, action)
        terminated = bool(self.goal_states and next_state in self.goal_states)
        info = {"valid": valid, "valid_actions": self.valid_actions(next_state)}
        return StepResult(next_state, float(terminated), terminated, False, valid, info)

    def rollout(self, start: State, actions: tuple[Action, ...]) -> tuple[State, bool]:
        state = start
        valid_path = True
        for action in actions:
            valid_path = valid_path and self.is_valid(state, action)
            state = self.step(state, action)
        return state, valid_path


def metadata(
    family: str,
    geometry: str,
    *,
    observation: str = "latent_state",
    memory_pressure: str = "unspecified",
    axes: dict[str, object] | None = None,
    tags: tuple[str, ...] = (),
) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        family=family,
        geometry=geometry,
        observation=observation,
        memory_pressure=memory_pressure,
        axes=tuple(sorted((axes or {}).items())),
        tags=tags,
    )


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
    return Environment(
        "torus_lattice",
        states,
        actions,
        transition,
        frozenset(transition),
        metadata=metadata(
            "lattice",
            "locally_metric_periodic",
            memory_pressure="low",
            axes={"commutativity": "high", "loop_closure": "exact", "boundary": "none"},
            tags=("metric", "commutative", "control"),
        ),
        start_states=((0, 0),),
    )


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
    return Environment(
        "clamped_lattice",
        states,
        actions,
        transition,
        _valid_from_nonself(transition),
        metadata=metadata(
            "lattice",
            "locally_metric_bounded",
            memory_pressure="low_boundary_sensitive",
            axes={"commutativity": "local_high_raw_medium", "loop_closure": "bounded", "boundary": "clamped"},
            tags=("metric", "boundary"),
        ),
        start_states=((0, 0),),
    )


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
    return Environment(
        "tree",
        tuple(states),
        actions,
        transition,
        _valid_from_nonself(transition),
        depth,
        metadata=metadata(
            "tree",
            "hierarchical_branching",
            memory_pressure="route_history",
            axes={"commutativity": "low", "branching": branching, "depth": depth},
            tags=("tree", "noncommutative"),
        ),
        start_states=((),),
    )


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
    return Environment(
        f"shortcut_tree_{suffix:03d}",
        states,
        actions,
        transition,
        frozenset(valid),
        depth,
        metadata=metadata(
            "shortcut_tree",
            "hierarchical_with_lateral_edges",
            memory_pressure="route_history",
            axes={
                "commutativity": "low_to_medium",
                "branching": branching,
                "depth": depth,
                "shortcut_density": shortcut_density,
            },
            tags=("tree", "shortcut", "mixed"),
        ),
        start_states=((),),
    )


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
    return Environment(
        "bottleneck_rooms",
        states,
        actions,
        transition,
        _valid_from_nonself(transition),
        metadata=metadata(
            "rooms",
            "local_metric_global_bottleneck",
            memory_pressure="mixed_local_global",
            axes={"commutativity": "local_high_raw_medium", "bottleneck": "single_door", "rooms": 2},
            tags=("rooms", "bottleneck", "mixed"),
        ),
        start_states=((0, 0, 0),),
    )


def topology_edges(topology: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    if topology == "loop_rich":
        return 4, ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
    if topology == "bottleneck":
        return 6, ((0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3))
    if topology == "tree_like":
        return 7, ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))
    raise ValueError(f"unknown topology: {topology}")


def door_state(room: int, port: str, size: int) -> State:
    mid = size // 2
    if port == "N":
        return (room, mid, size - 1)
    if port == "E":
        return (room, size - 1, mid)
    if port == "S":
        return (room, mid, 0)
    if port == "W":
        return (room, 0, mid)
    raise ValueError(port)


def room_graph_environment(topology: str, size: int = 3) -> Environment:
    """Rooms with local grid geometry and controlled global connectivity."""
    room_count, edges = topology_edges(topology)
    actions = ("N", "E", "S", "W")
    states = tuple((room, x, y) for room in range(room_count) for x in range(size) for y in range(size))
    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
    for room, x, y in states:
        candidates = {
            "N": (room, x, min(size - 1, y + 1)),
            "E": (room, min(size - 1, x + 1), y),
            "S": (room, x, max(0, y - 1)),
            "W": (room, max(0, x - 1), y),
        }
        for action, target in candidates.items():
            transition[((room, x, y), action)] = target
            if target != (room, x, y):
                valid.add(((room, x, y), action))

    neighbors: dict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    ports = ("N", "E", "S", "W")
    room_ports = {
        (room, neighbor): ports[idx]
        for room, room_neighbors in neighbors.items()
        for idx, neighbor in enumerate(sorted(room_neighbors))
    }
    for left, right in edges:
        left_port = room_ports[(left, right)]
        right_port = room_ports[(right, left)]
        left_door = door_state(left, left_port, size)
        right_door = door_state(right, right_port, size)
        transition[(left_door, left_port)] = right_door
        transition[(right_door, right_port)] = left_door
        valid.add((left_door, left_port))
        valid.add((right_door, right_port))

    center = (0, size // 2, size // 2)
    return Environment(
        f"room_graph_{topology}",
        states,
        actions,
        transition,
        frozenset(valid),
        metadata=metadata(
            "room_graph",
            topology,
            memory_pressure="mixed_local_global",
            axes={
                "rooms": room_count,
                "room_size": size,
                "global_edges": len(edges),
                "branching": max(len(room_neighbors) for room_neighbors in neighbors.values()),
            },
            tags=("rooms", "navigation", topology),
        ),
        start_states=(center,),
    )


def room_graph_family(size: int = 3) -> list[Environment]:
    return [
        room_graph_environment("loop_rich", size=size),
        room_graph_environment("bottleneck", size=size),
        room_graph_environment("tree_like", size=size),
    ]


def aliased_rooms(size: int = 3) -> Environment:
    """Two identical rooms whose local observations can hide room identity."""
    actions = ("N", "S", "E", "W")
    states = tuple((room, x, y) for room in range(2) for x in range(size) for y in range(size))
    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
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
    return Environment(
        "aliased_rooms",
        states,
        actions,
        transition,
        frozenset(valid),
        metadata=metadata(
            "aliased_navigation",
            "repeated_room_bottleneck",
            observation="local_position_aliases_room",
            memory_pressure="latent_context",
            axes={"rooms": 2, "room_size": size, "aliasing": "room_identity_hidden", "bottleneck": "single_door"},
            tags=("rooms", "aliased", "navigation"),
        ),
        start_states=((0, 0, 0),),
    )


def aliased_loop_rooms(size: int = 2, rooms: int = 3) -> Environment:
    """Ring of identical small rooms with hidden room identity."""
    actions = ("N", "S", "E", "W")
    states = tuple((room, x, y) for room in range(rooms) for x in range(size) for y in range(size))
    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
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
        east_door = (room, size - 1, 0)
        west_door = (room, 0, size - 1)
        transition[(east_door, "E")] = ((room + 1) % rooms, 0, 0)
        transition[(west_door, "W")] = ((room - 1) % rooms, size - 1, size - 1)
        valid.add((east_door, "E"))
        valid.add((west_door, "W"))
    return Environment(
        "aliased_loop_rooms",
        states,
        actions,
        transition,
        frozenset(valid),
        metadata=metadata(
            "aliased_navigation",
            "looped_repeated_rooms",
            observation="local_position_aliases_room",
            memory_pressure="latent_context_loop",
            axes={"rooms": rooms, "room_size": size, "aliasing": "room_identity_hidden", "loop_closure": "ring"},
            tags=("rooms", "aliased", "loop"),
        ),
        start_states=((0, 0, 0),),
    )


def aliased_t_maze(corridor_length: int = 8) -> Environment:
    """Cue-dependent T-maze where the cue is hidden after corridor entry."""
    actions = ("C0", "C1", "F", "B", "L", "R")
    left_arm = corridor_length
    right_arm = corridor_length + 1
    states: list[State] = [()]
    for cue in (0, 1):
        for pos in range(corridor_length + 2):
            states.append((cue, pos))
    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
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
    return Environment(
        "aliased_t_maze",
        tuple(states),
        actions,
        transition,
        frozenset(valid),
        metadata=metadata(
            "cue_gap",
            "t_maze",
            observation="cue_then_aliased_corridor",
            memory_pressure="delayed_cue",
            axes={"corridor_length": corridor_length, "cue_visibility": "initial_only", "branching": 2},
            tags=("aliased", "cue", "delay"),
        ),
        start_states=((),),
    )


def landmark_gap_detour() -> Environment:
    """Cue, sensory gap, forced detour, then route-dependent choice."""
    actions = ("C0", "C1", "F", "UP", "DOWN", "B", "L", "R")
    states: list[State] = [()]
    for cue in (0, 1):
        for stage in range(9):
            states.append((cue, stage))
    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
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
    return Environment(
        "landmark_gap_detour",
        tuple(states),
        actions,
        transition,
        frozenset(valid),
        metadata=metadata(
            "cue_gap",
            "detour_choice",
            observation="cue_then_gap_then_choice",
            memory_pressure="delayed_route_context",
            axes={"cue_visibility": "initial_only", "gap_stages": 5, "choice_arms": 2},
            tags=("cue", "delay", "route"),
        ),
        start_states=((),),
    )


def shortcut_route_choice() -> Environment:
    """Identical final junction after short or turning route histories."""
    actions = ("Q0", "Q1", "F", "TURN", "B", "CUT", "ARC")
    states: list[State] = [()]
    for route in (0, 1):
        for stage in range(8):
            states.append((route, stage))
    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
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
    return Environment(
        "shortcut_route_choice",
        tuple(states),
        actions,
        transition,
        frozenset(valid),
        metadata=metadata(
            "route_choice",
            "shortcut_vs_turning_route",
            observation="route_aliases_final_choice",
            memory_pressure="route_history",
            axes={"route_count": 2, "choice_arms": 2, "cue_visibility": "initial_only"},
            tags=("route", "choice", "aliased"),
        ),
        start_states=((),),
    )


def behavioral_navigation_family() -> list[Environment]:
    return [
        aliased_rooms(),
        aliased_t_maze(),
        landmark_gap_detour(),
        shortcut_route_choice(),
        aliased_loop_rooms(),
    ]


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
    return Environment(
        f"trace_commute_{suffix:03d}",
        tuple(sorted(states_set)),
        actions,
        transition,
        frozenset(valid),
        max_depth,
        metadata=metadata(
            "trace_monoid",
            "action_history_quotient",
            memory_pressure="order_history" if suffix < 100 else "count_state",
            axes={
                "commutativity_density": len(symmetric_pairs) / total_pairs if total_pairs else 1.0,
                "max_depth": max_depth,
                "alphabet_size": len(actions),
            },
            tags=("trace", "formal", "history_compressibility"),
        ),
        start_states=((),),
    )


def trace_monoid_family(max_depth: int = 8) -> list[Environment]:
    return [
        trace_monoid(frozenset(), max_depth=max_depth),
        trace_monoid(frozenset({("A", "B")}), max_depth=max_depth),
        trace_monoid(frozenset({("A", "B"), ("A", "C")}), max_depth=max_depth),
        trace_monoid(frozenset({("A", "B"), ("A", "C"), ("B", "C")}), max_depth=max_depth),
    ]


def trace_monoid_general(
    action_count: int,
    commuting_pairs: frozenset[tuple[Action, Action]],
    max_depth: int,
    *,
    relation_id: int = 0,
) -> Environment:
    """Trace-monoid constructor for formal sweeps with variable alphabet size."""
    actions = tuple(chr(ord("A") + idx) for idx in range(action_count))
    action_rank = {action: idx for idx, action in enumerate(actions)}
    symmetric_pairs = frozenset(
        tuple(sorted(pair, key=action_rank.__getitem__)) for pair in commuting_pairs
    )

    def commutes(first: Action, second: Action) -> bool:
        return first == second or tuple(sorted((first, second), key=action_rank.__getitem__)) in symmetric_pairs

    def canonicalize(word: tuple[Action, ...]) -> State:
        ordered = list(word)
        changed = True
        while changed:
            changed = False
            for idx in range(len(ordered) - 1):
                if action_rank[ordered[idx]] > action_rank[ordered[idx + 1]] and commutes(ordered[idx], ordered[idx + 1]):
                    ordered[idx], ordered[idx + 1] = ordered[idx + 1], ordered[idx]
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
                states_set.add(target)
                next_frontier.add(target)
        frontier = next_frontier

    transition: Transition = {}
    valid: set[tuple[State, Action]] = set()
    for state in states_set:
        word = tuple(actions[idx] for idx in state)
        for action in actions:
            if len(state) >= max_depth:
                transition[(state, action)] = state
            else:
                transition[(state, action)] = canonicalize(word + (action,))
                valid.add((state, action))

    total_pairs = action_count * (action_count - 1) // 2
    density = len(symmetric_pairs) / total_pairs if total_pairs else 1.0
    suffix = int(round(100 * density))
    name = f"trace_k{action_count}_d{max_depth}_rho{suffix:03d}_r{relation_id}"
    return Environment(
        name,
        tuple(sorted(states_set)),
        actions,
        transition,
        frozenset(valid),
        max_depth,
        metadata=metadata(
            "trace_monoid",
            "variable_action_history_quotient",
            memory_pressure="order_history" if suffix < 100 else "count_state",
            axes={
                "commutativity_density": density,
                "max_depth": max_depth,
                "alphabet_size": action_count,
                "relation_id": relation_id,
            },
            tags=("trace", "formal", "sweep"),
        ),
        start_states=((),),
    )


def trace_env(commuting: frozenset[tuple[Action, Action]], name_suffix: str, max_depth: int = 6) -> Environment:
    """Backward-compatible named trace environment used by navigation benchmarks."""
    env = trace_monoid(commuting, max_depth=max_depth)
    return Environment(
        f"trace_{name_suffix}",
        env.states,
        env.actions,
        env.transition,
        env.valid,
        env.max_depth,
        env.metadata,
        env.start_states,
        env.goal_states,
    )


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


def graph_edges(env: Environment, *, valid_only: bool = True) -> list[tuple[State, Action, State]]:
    edges = []
    for state in env.states:
        for action in env.actions:
            if valid_only and not env.is_valid(state, action):
                continue
            edges.append((state, action, env.step(state, action)))
    return edges


def validate_environment(env: Environment) -> list[str]:
    errors: list[str] = []
    state_set = set(env.states)
    action_set = set(env.actions)
    if len(state_set) != len(env.states):
        errors.append("duplicate states")
    if len(action_set) != len(env.actions):
        errors.append("duplicate actions")
    for state, action in env.transition:
        if state not in state_set:
            errors.append(f"transition references unknown state: {state}")
        if action not in action_set:
            errors.append(f"transition references unknown action: {action}")
    for target in env.transition.values():
        if target not in state_set:
            errors.append(f"transition targets unknown state: {target}")
    for state, action in env.valid:
        if state not in state_set:
            errors.append(f"valid mask references unknown state: {state}")
        if action not in action_set:
            errors.append(f"valid mask references unknown action: {action}")
    for start in env.start_states:
        if start not in state_set:
            errors.append(f"unknown start state: {start}")
    for goal in env.goal_states:
        if goal not in state_set:
            errors.append(f"unknown goal state: {goal}")
    return errors


def environment_summary(env: Environment) -> dict[str, object]:
    valid_edges = graph_edges(env, valid_only=True)
    self_loops = sum(1 for state, _, target in graph_edges(env, valid_only=False) if state == target)
    return {
        "name": env.name,
        "family": env.metadata.family,
        "geometry": env.metadata.geometry,
        "n_states": len(env.states),
        "n_actions": len(env.actions),
        "n_valid_transitions": len(valid_edges),
        "n_self_loops": self_loops,
        "max_depth": env.max_depth,
        "axes": dict(env.metadata.axes),
        "tags": env.metadata.tags,
    }


def action_counts(state: State, n_actions: int | None = None) -> tuple[int, ...]:
    if n_actions is None:
        n_actions = max(state) + 1 if state else 0
    counts = [0] * n_actions
    for action in state:
        if action >= n_actions:
            counts.extend(0 for _ in range(action - n_actions + 1))
            n_actions = action + 1
        counts[action] += 1
    return tuple(counts)


def observe_trace_state(state: State, mode: str, *, n_actions: int = 3) -> Observation:
    if mode == "full":
        return state
    if mode == "counts":
        return action_counts(state, n_actions)
    if mode == "depth_last":
        return (len(state), state[-1] if state else -1)
    if mode == "depth":
        return len(state)
    raise ValueError(f"unknown observation mode: {mode}")


def observe_room_state(state: State, mode: str) -> Observation:
    """Observation summaries for room states shaped as ``(room, x, y)``."""
    if len(state) != 3:
        raise ValueError(f"room observation expects a 3D room state, got {state}")
    room, x, y = state
    if mode == "full":
        return state
    if mode == "room":
        return room
    if mode == "local_position":
        return (x, y)
    if mode == "room_local_position":
        return (room, x, y)
    if mode == "boundary_signature":
        return (x == 0, y == 0)
    raise ValueError(f"unknown room observation mode: {mode}")


def observe_behavior_state(state: State, env_name: str, mode: str) -> Observation:
    """Symbolic observation summaries for aliased and cue-gap navigation tasks."""
    if mode == "full":
        return state
    if mode == "cue_only":
        return state[0] if state else -1
    if mode == "stage_only":
        return state[1] if len(state) > 1 else -1
    if mode == "local_symbol":
        return sparse_observation_symbol(state, env_name)
    if mode == "cue_then_blank":
        if state == ():
            return ("root",)
        cue, stage = state[:2]
        return (cue, stage) if stage == 0 else ("blank", stage)
    raise ValueError(f"unknown behavior observation mode: {mode}")


def sparse_observation_symbol(state: State, env_name: str) -> int:
    if env_name.startswith("trace"):
        return (state[-1] if state else 3) % 4
    if env_name in {"aliased_rooms", "aliased_loop_rooms"} or env_name.startswith("room_graph_"):
        _, x, y = state
        return (x + 2 * y) % 5
    if env_name in {"aliased_t_maze", "long_aliased_t_maze"}:
        if state == ():
            return 0
        cue, pos = state
        if pos == 0:
            return 1 + cue
        return 5 if pos >= 8 else 4
    if env_name == "landmark_gap_detour":
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
    if env_name == "shortcut_route_choice":
        if state == ():
            return 0
        route, stage = state
        if stage == 0:
            return 1 + route
        if stage == 5:
            return 5
        return 4
    if env_name in {"torus_lattice", "clamped_lattice"}:
        x, y = state
        return (x + y) % 4
    if env_name == "bottleneck_rooms":
        _, x, y = state
        return (x + y) % 4
    return (len(state), state[-1] if state else 0)[-1] % 4


def numeric_state_observation(state: State, *, width: int = 6) -> tuple[float, ...]:
    """Coordinate-like fixed-width observation suitable for richer simulators."""
    values = [float(value) for value in state[:width]]
    values += [0.0] * (width - len(values))
    scale = max(1.0, max((abs(value) for value in values), default=1.0))
    return tuple(value / scale for value in values)


def local_view_observation(env: Environment, state: State) -> tuple[float, ...]:
    """High-dimensional local affordance vector: valid action bits plus sparse symbol."""
    action_bits = tuple(1.0 if env.is_valid(state, action) else 0.0 for action in env.actions)
    symbol = sparse_observation_symbol(state, env.name)
    symbol_bits = tuple(1.0 if idx == symbol % 6 else 0.0 for idx in range(6))
    return action_bits + symbol_bits


def sensory_alias_key(env: Environment, state: State) -> tuple[object, ...]:
    """Current-state sensory class for realistic noisy observation channels.

    The key contains no action, action-history, or valid-action affordance
    information. It maps latent state to locally observable sensory content,
    intentionally aliasing hidden room, route, or cue variables where relevant.
    """
    family = env.metadata.family
    if state == ():
        return ("root",)
    if len(state) == 3 and (
        family in {"rooms", "room_graph", "aliased_navigation"} or env.name.startswith("room_graph_")
    ):
        _, x, y = state
        return ("room_local", x, y)
    if family in {"cue_gap", "route_choice"}:
        cue_or_route, stage = state[:2]
        if stage == 0:
            return ("initial_cue", cue_or_route)
        if env.name == "landmark_gap_detour":
            if stage in {2, 3, 4}:
                return ("gap",)
            if stage >= 6:
                return ("choice", stage)
            return ("corridor", stage)
        if env.name == "shortcut_route_choice":
            return ("choice", stage) if stage == 5 else ("route_local", stage)
        if env.name in {"aliased_t_maze", "long_aliased_t_maze"}:
            max_pos = max((candidate[1] for candidate in env.states if candidate), default=stage)
            if stage >= max_pos - 1:
                return ("arm", stage)
            return ("corridor", stage)
        return ("behavior_local", stage)
    if family == "trace_monoid" or env.name.startswith("trace"):
        return ("trace_depth_last", len(state), state[-1] if state else -1)
    if family in {"lattice", "tree", "shortcut_tree"}:
        return (family, *state)
    return ("symbol", sparse_observation_symbol(state, env.name))


def _state_seed(seed: int | None, state: State, salt: str) -> int:
    text = f"{seed!r}|{salt}|{state!r}"
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _one_hot(index: int, size: int) -> tuple[float, ...]:
    return tuple(1.0 if idx == index % size else 0.0 for idx in range(size))


def _prototype_vector(key: tuple[object, ...], *, dim: int, seed: int, salt: str) -> tuple[float, ...]:
    rng = Random(_state_seed(seed, tuple(ord(char) for char in repr(key)), salt))
    return tuple(rng.uniform(-1.0, 1.0) for _ in range(dim))


def gaussian_local_state_channel(env: Environment, sigma: float, *, width: int = 6) -> ObservationChannel:
    """Coordinate-like state vector with independent Gaussian observation noise."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    def observe(state: State, seed: int | None = None) -> tuple[float, ...]:
        base = numeric_state_observation(state, width=width)
        if sigma == 0.0:
            return base
        rng = Random(_state_seed(seed, state, f"gaussian:{env.name}:{sigma}:{width}"))
        return tuple(value + rng.gauss(0.0, sigma) for value in base)

    return ObservationChannel(
        name=f"{env.name}_gaussian_local_sigma_{sigma:g}",
        kind="gaussian_local_state",
        dim=width,
        observe=observe,
        metadata=tuple(sorted({
            "env": env.name,
            "noise_sigma": sigma,
            "width": width,
        }.items())),
    )


def gaussian_sensory_channel(
    env: Environment,
    sigma: float = 0.1,
    *,
    dim: int = 8,
    seed: int = 0,
) -> ObservationChannel:
    """Aliased current-state sensory prototype plus independent Gaussian noise."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if dim <= 0:
        raise ValueError("dim must be positive")
    prototype_seed = seed

    def observe(state: State, seed: int | None = None) -> tuple[float, ...]:
        key = sensory_alias_key(env, state)
        base = _prototype_vector(key, dim=dim, seed=prototype_seed, salt=f"sensory:{env.name}:{dim}")
        if sigma == 0.0:
            return base
        rng = Random(_state_seed(seed, state, f"gaussian_sensory:{env.name}:{sigma}:{dim}:{prototype_seed}"))
        return tuple(value + rng.gauss(0.0, sigma) for value in base)

    return ObservationChannel(
        name=f"{env.name}_gaussian_sensory_sigma_{sigma:g}",
        kind="gaussian_sensory",
        dim=dim,
        observe=observe,
        metadata=tuple(sorted({
            "env": env.name,
            "noise_sigma": sigma,
            "dim": dim,
            "prototype_seed": prototype_seed,
            "includes_actions": False,
        }.items())),
    )


def sparse_visual_feature_channel(
    env: Environment,
    *,
    n_features: int = 6,
    alias_rooms: bool = True,
    dropout_p: float = 0.0,
) -> ObservationChannel:
    """Sparse visual cue vector with optional dropout and intentional room aliasing."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if not 0.0 <= dropout_p <= 1.0:
        raise ValueError("dropout_p must be in [0, 1]")

    def symbol(state: State) -> int:
        if alias_rooms and len(state) == 3:
            _, x, y = state
            return (x + 2 * y) % n_features
        return sparse_observation_symbol(state, env.name) % n_features

    def observe(state: State, seed: int | None = None) -> tuple[float, ...]:
        rng = Random(_state_seed(seed, state, f"sparse:{env.name}:{n_features}:{alias_rooms}:{dropout_p}"))
        if dropout_p and rng.random() < dropout_p:
            return (0.0,) * n_features
        return _one_hot(symbol(state), n_features)

    return ObservationChannel(
        name=f"{env.name}_sparse_visual_{n_features}",
        kind="sparse_visual_features",
        dim=n_features,
        observe=observe,
        metadata=tuple(sorted({
            "env": env.name,
            "n_features": n_features,
            "alias_rooms": alias_rooms,
            "dropout_p": dropout_p,
        }.items())),
    )


def egocentric_noisy_landmark_channel(
    env: Environment,
    *,
    sigma: float,
    n_landmarks: int = 6,
) -> ObservationChannel:
    """Valid-action affordance bits plus noisy landmark-like sensory features."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if n_landmarks <= 0:
        raise ValueError("n_landmarks must be positive")
    base_sparse = sparse_visual_feature_channel(env, n_features=n_landmarks, alias_rooms=True, dropout_p=0.0)

    def observe(state: State, seed: int | None = None) -> tuple[float, ...]:
        action_bits = tuple(1.0 if env.is_valid(state, action) else 0.0 for action in env.actions)
        landmarks = base_sparse.observe(state, seed)
        if sigma == 0.0:
            return action_bits + landmarks
        rng = Random(_state_seed(seed, state, f"egocentric:{env.name}:{sigma}:{n_landmarks}"))
        noisy_landmarks = tuple(value + rng.gauss(0.0, sigma) for value in landmarks)
        return action_bits + noisy_landmarks

    return ObservationChannel(
        name=f"{env.name}_egocentric_landmarks_sigma_{sigma:g}",
        kind="egocentric_noisy_landmarks",
        dim=len(env.actions) + n_landmarks,
        observe=observe,
        metadata=tuple(sorted({
            "env": env.name,
            "noise_sigma": sigma,
            "n_landmarks": n_landmarks,
            "egocentric": True,
        }.items())),
    )


def quantized_observation(observation: tuple[float, ...], *, bin_width: float = 0.25) -> tuple[int, ...]:
    if bin_width <= 0.0:
        raise ValueError("bin_width must be positive")
    return tuple(round(value / bin_width) for value in observation)


def observation_ambiguity(env: Environment, observe: ObservationFn) -> dict[str, float]:
    groups: dict[Observation, list[State]] = defaultdict(list)
    for state in env.states:
        groups[observe(state)].append(state)

    n_states = len(env.states)
    weighted_sizes = [len(groups[observe(state)]) for state in env.states]
    mean_candidates = sum(weighted_sizes) / n_states
    max_candidates = max(weighted_sizes)
    normalized = 0.0 if n_states <= 1 else (mean_candidates - 1) / (n_states - 1)
    obs_counts = {obs: len(states) for obs, states in groups.items()}
    entropy = -sum((count / n_states) * log2(count / n_states) for count in obs_counts.values())
    max_entropy = log2(n_states) if n_states > 1 else 0.0
    retained_fraction = entropy / max_entropy if max_entropy else 1.0
    return {
        "n_states": float(n_states),
        "n_observations": float(len(groups)),
        "mean_candidates": mean_candidates,
        "max_candidates": float(max_candidates),
        "normalized_ambiguity": normalized,
        "retained_entropy_fraction": retained_fraction,
    }


def transition_ambiguity(env: Environment, observe: ObservationFn) -> dict[str, float]:
    outcomes: dict[tuple[Observation, Action], set[State]] = defaultdict(set)
    weights: dict[tuple[Observation, Action], int] = defaultdict(int)

    for state in env.states:
        obs = observe(state)
        for action in env.actions:
            if not env.is_valid(state, action):
                continue
            key = (obs, action)
            outcomes[key].add(env.step(state, action))
            weights[key] += 1

    if not outcomes:
        return {
            "mean_next_candidates": float("nan"),
            "max_next_candidates": float("nan"),
            "ambiguous_fraction": float("nan"),
            "weighted_ambiguous_fraction": float("nan"),
        }

    total_weight = sum(weights.values())
    mean_next = sum(len(outcomes[key]) * weights[key] for key in outcomes) / total_weight
    max_next = max(len(values) for values in outcomes.values())
    ambiguous_keys = [key for key, values in outcomes.items() if len(values) > 1]
    return {
        "mean_next_candidates": mean_next,
        "max_next_candidates": float(max_next),
        "ambiguous_fraction": len(ambiguous_keys) / len(outcomes),
        "weighted_ambiguous_fraction": sum(weights[key] for key in ambiguous_keys) / total_weight,
    }


def reversible_navigation_graph(env: Environment) -> dict[State, list[tuple[str, State]]]:
    graph: dict[State, list[tuple[str, State]]] = defaultdict(list)
    for state in env.states:
        for action in env.actions:
            if not env.is_valid(state, action):
                continue
            next_state = env.step(state, action)
            graph[state].append((action, next_state))
            graph[next_state].append((f"undo_{action}", state))
    return graph


def default_goal_state(env: Environment) -> State:
    if env.goal_states:
        return env.goal_states[0]
    if () in env.states and env.actions:
        length = min(8, env.max_depth or 8)
        sequence = tuple(env.actions[idx % len(env.actions)] for idx in range(length))
        goal, valid = env.rollout((), sequence)
        if valid:
            return goal
    return env.states[-1]


def distances_to_goal(env: Environment, goal: State) -> dict[State, int]:
    graph = reversible_navigation_graph(env)
    distances = {goal: 0}
    queue: deque[State] = deque([goal])
    while queue:
        state = queue.popleft()
        for _, neighbor in graph[state]:
            if neighbor not in distances:
                distances[neighbor] = distances[state] + 1
                queue.append(neighbor)
    return distances


def optimal_policy_sets(env: Environment, goal: State | None = None) -> dict[State, Policy]:
    goal = goal if goal is not None else default_goal_state(env)
    graph = reversible_navigation_graph(env)
    distances = distances_to_goal(env, goal)
    policies: dict[State, Policy] = {}
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


def policy_ambiguity(env: Environment, observe: ObservationFn, goal: State | None = None) -> dict[str, float]:
    goal = goal if goal is not None else default_goal_state(env)
    policies = optimal_policy_sets(env, goal)
    by_obs: dict[Observation, list[Policy]] = defaultdict(list)
    for state, policy in policies.items():
        by_obs[observe(state)].append(policy)

    if not policies:
        return {
            "goal_depth": float(len(goal)),
            "n_policy_states": 0.0,
            "policy_ambiguous_observation_fraction": float("nan"),
            "weighted_policy_ambiguity": float("nan"),
            "mean_policy_set_size": float("nan"),
        }

    ambiguous_obs = {
        obs for obs, policy_list in by_obs.items()
        if len(set(policy_list)) > 1
    }
    return {
        "goal_depth": float(len(goal)),
        "n_policy_states": float(len(policies)),
        "policy_ambiguous_observation_fraction": len(ambiguous_obs) / len(by_obs),
        "weighted_policy_ambiguity": sum(
            len(policy_list) for obs, policy_list in by_obs.items()
            if obs in ambiguous_obs
        ) / len(policies),
        "mean_policy_set_size": sum(len(policy) for policy in policies.values()) / len(policies),
    }


def residual_history_diagnostics(env: Environment, observe: ObservationFn, goal: State | None = None) -> dict[str, float]:
    """Summarize whether an observation leaves transition or policy-relevant history."""
    transition = transition_ambiguity(env, observe)
    policy = policy_ambiguity(env, observe, goal)
    transition_score = transition["weighted_ambiguous_fraction"]
    policy_score = policy["weighted_policy_ambiguity"]
    finite_scores = [value for value in (transition_score, policy_score) if value == value]
    residual = max(finite_scores) if finite_scores else float("nan")
    return {
        "transition_residual": transition_score,
        "policy_residual": policy_score,
        "residual_history_score": residual,
    }


def sampled_observation_ambiguity(
    env: Environment,
    channel: ObservationChannel,
    *,
    n_samples: int = 1,
    seed: int = 1,
    bin_width: float = 0.25,
) -> dict[str, float]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    groups: dict[tuple[int, ...], set[State]] = defaultdict(set)
    state_bins: dict[State, set[tuple[int, ...]]] = defaultdict(set)
    for sample_idx in range(n_samples):
        for state in env.states:
            obs_seed = _state_seed(seed + sample_idx, state, channel.name)
            obs = channel.observe(state, obs_seed)
            key = quantized_observation(obs, bin_width=bin_width)
            groups[key].add(state)
            state_bins[state].add(key)

    weighted_sizes = []
    for state, bins in state_bins.items():
        sizes = [len(groups[key]) for key in bins]
        weighted_sizes.append(sum(sizes) / len(sizes))
    n_states = len(env.states)
    mean_candidates = sum(weighted_sizes) / n_states if n_states else float("nan")
    max_candidates = max((len(states) for states in groups.values()), default=float("nan"))
    normalized = 0.0 if n_states <= 1 else (mean_candidates - 1) / (n_states - 1)
    collision_bins = sum(1 for states in groups.values() if len(states) > 1)
    return {
        "n_states": float(n_states),
        "n_observation_bins": float(len(groups)),
        "mean_candidates": mean_candidates,
        "max_candidates": float(max_candidates),
        "normalized_ambiguity": normalized,
        "collision_bin_fraction": collision_bins / len(groups) if groups else float("nan"),
    }


def sampled_transition_ambiguity(
    env: Environment,
    channel: ObservationChannel,
    *,
    n_samples: int = 1,
    seed: int = 1,
    bin_width: float = 0.25,
) -> dict[str, float]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    outcomes: dict[tuple[tuple[int, ...], Action], set[State]] = defaultdict(set)
    weights: dict[tuple[tuple[int, ...], Action], int] = defaultdict(int)
    for sample_idx in range(n_samples):
        for state in env.states:
            obs_seed = _state_seed(seed + sample_idx, state, channel.name)
            obs = quantized_observation(channel.observe(state, obs_seed), bin_width=bin_width)
            for action in env.actions:
                if not env.is_valid(state, action):
                    continue
                key = (obs, action)
                outcomes[key].add(env.step(state, action))
                weights[key] += 1

    if not outcomes:
        return {
            "mean_next_candidates": float("nan"),
            "max_next_candidates": float("nan"),
            "ambiguous_fraction": float("nan"),
            "weighted_ambiguous_fraction": float("nan"),
        }
    total_weight = sum(weights.values())
    ambiguous_keys = [key for key, values in outcomes.items() if len(values) > 1]
    return {
        "mean_next_candidates": sum(len(outcomes[key]) * weights[key] for key in outcomes) / total_weight,
        "max_next_candidates": float(max(len(values) for values in outcomes.values())),
        "ambiguous_fraction": len(ambiguous_keys) / len(outcomes),
        "weighted_ambiguous_fraction": sum(weights[key] for key in ambiguous_keys) / total_weight,
    }


def sampled_policy_ambiguity(
    env: Environment,
    channel: ObservationChannel,
    *,
    goal: State | None = None,
    n_samples: int = 1,
    seed: int = 1,
    bin_width: float = 0.25,
) -> dict[str, float]:
    goal = goal if goal is not None else default_goal_state(env)
    policies = optimal_policy_sets(env, goal)
    by_obs: dict[tuple[int, ...], list[Policy]] = defaultdict(list)
    for sample_idx in range(n_samples):
        for state, policy in policies.items():
            obs_seed = _state_seed(seed + sample_idx, state, channel.name)
            obs = quantized_observation(channel.observe(state, obs_seed), bin_width=bin_width)
            by_obs[obs].append(policy)

    if not policies:
        return {
            "goal_depth": float(len(goal)),
            "n_policy_states": 0.0,
            "policy_ambiguous_observation_fraction": float("nan"),
            "weighted_policy_ambiguity": float("nan"),
            "mean_policy_set_size": float("nan"),
        }
    ambiguous_obs = {
        obs for obs, policy_list in by_obs.items()
        if len(set(policy_list)) > 1
    }
    total_policy_samples = len(policies) * n_samples
    return {
        "goal_depth": float(len(goal)),
        "n_policy_states": float(len(policies)),
        "policy_ambiguous_observation_fraction": len(ambiguous_obs) / len(by_obs) if by_obs else float("nan"),
        "weighted_policy_ambiguity": sum(
            len(policy_list) for obs, policy_list in by_obs.items()
            if obs in ambiguous_obs
        ) / total_policy_samples,
        "mean_policy_set_size": sum(len(policy) for policy in policies.values()) / len(policies),
    }


def sampled_residual_history_diagnostics(
    env: Environment,
    channel: ObservationChannel,
    *,
    goal: State | None = None,
    n_samples: int = 1,
    seed: int = 1,
    bin_width: float = 0.25,
) -> dict[str, float]:
    transition = sampled_transition_ambiguity(env, channel, n_samples=n_samples, seed=seed, bin_width=bin_width)
    policy = sampled_policy_ambiguity(env, channel, goal=goal, n_samples=n_samples, seed=seed, bin_width=bin_width)
    transition_score = transition["weighted_ambiguous_fraction"]
    policy_score = policy["weighted_policy_ambiguity"]
    finite_scores = [value for value in (transition_score, policy_score) if value == value]
    return {
        "transition_residual": transition_score,
        "policy_residual": policy_score,
        "residual_history_score": max(finite_scores) if finite_scores else float("nan"),
    }


def standard_observation_specs(env: Environment) -> list[ObservationSpec]:
    """Return comparable observation summaries for the current environment family."""
    if env.metadata.family == "trace_monoid" or env.name.startswith("trace"):
        return [
            ObservationSpec("full", lambda state: observe_trace_state(state, "full", n_actions=len(env.actions)), "Latent trace state."),
            ObservationSpec("counts", lambda state: observe_trace_state(state, "counts", n_actions=len(env.actions)), "Coordinate/count-like action summary."),
            ObservationSpec("depth_last", lambda state: observe_trace_state(state, "depth_last", n_actions=len(env.actions)), "Finite local history proxy."),
            ObservationSpec("depth", lambda state: observe_trace_state(state, "depth", n_actions=len(env.actions)), "Depth-only compression."),
        ]
    if env.metadata.family in {"rooms", "room_graph", "aliased_navigation"} or env.name.startswith("room_graph_"):
        return [
            ObservationSpec("full", lambda state: state, "Latent state."),
            ObservationSpec("local_position", lambda state: observe_room_state(state, "local_position"), "Room-local coordinate with room identity hidden."),
            ObservationSpec("local_symbol", lambda state: sparse_observation_symbol(state, env.name), "Sparse aliased local symbol."),
            ObservationSpec("local_view", lambda state: local_view_observation(env, state), "Affordance bits plus sparse local symbol."),
        ]
    if env.metadata.family in {"cue_gap", "route_choice"}:
        return [
            ObservationSpec("full", lambda state: state, "Latent state."),
            ObservationSpec("local_symbol", lambda state: observe_behavior_state(state, env.name, "local_symbol"), "Sparse landmark-like symbol."),
            ObservationSpec("stage_only", lambda state: observe_behavior_state(state, env.name, "stage_only"), "Progress coordinate without cue or route identity."),
            ObservationSpec("cue_then_blank", lambda state: observe_behavior_state(state, env.name, "cue_then_blank"), "Initial cue followed by blank/gap observations."),
        ]
    return [
        ObservationSpec("full", lambda state: state, "Latent state."),
        ObservationSpec("numeric", lambda state: numeric_state_observation(state), "Coordinate-like numeric summary."),
        ObservationSpec("sparse_symbol", lambda state: sparse_observation_symbol(state, env.name), "Sparse aliased symbol."),
    ]


def architecture_expectations(residual_history_score: float) -> dict[str, str]:
    """Heuristic acceptance expectations for model classes from residual history demand."""
    if residual_history_score != residual_history_score:
        return {
            "vector_count": "unknown",
            "finite_history": "unknown",
            "sequence": "unknown",
            "reservoir": "unknown",
            "hybrid": "unknown",
        }
    if residual_history_score <= 0.05:
        return {
            "vector_count": "expected_sufficient",
            "finite_history": "not_required",
            "sequence": "not_required",
            "reservoir": "not_required",
            "hybrid": "not_required",
        }
    if residual_history_score <= 0.30:
        return {
            "vector_count": "expected_partial",
            "finite_history": "expected_helpful",
            "sequence": "expected_helpful",
            "reservoir": "span_sensitive",
            "hybrid": "expected_helpful",
        }
    return {
        "vector_count": "expected_insufficient",
        "finite_history": "span_matched_needed",
        "sequence": "expected_useful",
        "reservoir": "span_and_training_sensitive",
        "hybrid": "expected_useful",
    }


def standard_diagnostic_table(envs: list[Environment] | None = None) -> list[dict[str, object]]:
    """Build a comparable diagnostic table across the standard taxonomy."""
    envs = envs or default_environments()
    rows: list[dict[str, object]] = []
    for env in envs:
        raw_commutativity = effective_commutativity(env, length=3, n_sequences=80, seed=7)
        valid_commutativity = effective_commutativity(env, length=3, n_sequences=80, seed=7, valid_only=True)
        summary = environment_summary(env)
        for spec in standard_observation_specs(env):
            observation = observation_ambiguity(env, spec.observe)
            transition = transition_ambiguity(env, spec.observe)
            policy = policy_ambiguity(env, spec.observe)
            residual = residual_history_diagnostics(env, spec.observe)
            expectations = architecture_expectations(residual["residual_history_score"])
            rows.append({
                "environment": env.name,
                "family": summary["family"],
                "geometry": summary["geometry"],
                "observation_mode": spec.name,
                "n_states": summary["n_states"],
                "n_actions": summary["n_actions"],
                "raw_commutativity": raw_commutativity,
                "valid_commutativity": valid_commutativity,
                **{f"observation_{key}": value for key, value in observation.items()},
                **{f"transition_{key}": value for key, value in transition.items()},
                **{f"policy_{key}": value for key, value in policy.items()},
                **residual,
                **{f"expect_{key}": value for key, value in expectations.items()},
            })
    return rows


def standard_sensory_diagnostic_table(
    envs: list[Environment] | None = None,
    *,
    sigma: float = 0.1,
    dim: int = 8,
    n_samples: int = 3,
    seed: int = 1,
    bin_width: float = 0.25,
) -> list[dict[str, object]]:
    """Build Gaussian-sensory sampled diagnostics across the standard taxonomy."""
    envs = envs or default_environments()
    rows: list[dict[str, object]] = []
    for env in envs:
        raw_commutativity = effective_commutativity(env, length=3, n_sequences=80, seed=7)
        valid_commutativity = effective_commutativity(env, length=3, n_sequences=80, seed=7, valid_only=True)
        summary = environment_summary(env)
        channel = gaussian_sensory_channel(env, sigma=sigma, dim=dim, seed=seed)
        observation = sampled_observation_ambiguity(
            env,
            channel,
            n_samples=n_samples,
            seed=seed,
            bin_width=bin_width,
        )
        transition = sampled_transition_ambiguity(
            env,
            channel,
            n_samples=n_samples,
            seed=seed,
            bin_width=bin_width,
        )
        policy = sampled_policy_ambiguity(
            env,
            channel,
            n_samples=n_samples,
            seed=seed,
            bin_width=bin_width,
        )
        residual = sampled_residual_history_diagnostics(
            env,
            channel,
            n_samples=n_samples,
            seed=seed,
            bin_width=bin_width,
        )
        expectations = architecture_expectations(residual["residual_history_score"])
        rows.append({
            "environment": env.name,
            "family": summary["family"],
            "geometry": summary["geometry"],
            "observation_mode": channel.kind,
            "observation_channel": channel.name,
            "n_states": summary["n_states"],
            "n_actions": summary["n_actions"],
            "raw_commutativity": raw_commutativity,
            "valid_commutativity": valid_commutativity,
            **{f"observation_{key}": value for key, value in observation.items()},
            **{f"transition_{key}": value for key, value in transition.items()},
            **{f"policy_{key}": value for key, value in policy.items()},
            **residual,
            **{f"expect_{key}": value for key, value in expectations.items()},
        })
    return rows


def trace_diagnostic_table(
    envs: list[Environment] | None = None,
    modes: tuple[str, ...] = ("full", "counts", "depth_last", "depth"),
) -> list[dict[str, object]]:
    envs = envs or trace_monoid_family(max_depth=8)
    rows: list[dict[str, object]] = []
    for env in envs:
        for mode in modes:
            observe = lambda state, mode=mode: observe_trace_state(state, mode, n_actions=len(env.actions))
            rows.append({
                "environment": env.name,
                "observation_mode": mode,
                **observation_ambiguity(env, observe),
                **{f"transition_{key}": value for key, value in transition_ambiguity(env, observe).items()},
                **{f"policy_{key}": value for key, value in policy_ambiguity(env, observe).items()},
            })
    return rows


def registered_environment_specs() -> list[EnvironmentSpec]:
    return [
        EnvironmentSpec("torus_lattice", "lattice", torus_lattice, description="Periodic local metric control."),
        EnvironmentSpec("clamped_lattice", "lattice", clamped_lattice, description="Bounded local metric grid."),
        EnvironmentSpec("bottleneck_rooms", "rooms", bottleneck_rooms, description="Two rooms linked by a single door."),
        EnvironmentSpec("tree", "tree", tree, description="Hierarchical noncommutative control."),
        EnvironmentSpec("shortcut_tree_025", "shortcut_tree", lambda: shortcut_tree(shortcut_density=0.25)),
        EnvironmentSpec("shortcut_tree_050", "shortcut_tree", lambda: shortcut_tree(shortcut_density=0.50)),
        EnvironmentSpec("shortcut_tree_100", "shortcut_tree", lambda: shortcut_tree(shortcut_density=1.00)),
        EnvironmentSpec("room_graph_loop_rich", "room_graph", lambda: room_graph_environment("loop_rich")),
        EnvironmentSpec("room_graph_bottleneck", "room_graph", lambda: room_graph_environment("bottleneck")),
        EnvironmentSpec("room_graph_tree_like", "room_graph", lambda: room_graph_environment("tree_like")),
        EnvironmentSpec("aliased_rooms", "aliased_navigation", aliased_rooms),
        EnvironmentSpec("aliased_t_maze", "cue_gap", aliased_t_maze),
        EnvironmentSpec("landmark_gap_detour", "cue_gap", landmark_gap_detour),
        EnvironmentSpec("shortcut_route_choice", "route_choice", shortcut_route_choice),
        EnvironmentSpec("aliased_loop_rooms", "aliased_navigation", aliased_loop_rooms),
        EnvironmentSpec("trace_monoid_family", "trace_monoid", lambda: trace_monoid_family()[0], description="Use trace_monoid_family() for the full formal sweep."),
    ]


def default_environments() -> list[Environment]:
    return [
        torus_lattice(),
        clamped_lattice(),
        bottleneck_rooms(),
        tree(branching=3, depth=4),
        shortcut_tree(branching=3, depth=4, shortcut_density=0.25),
        shortcut_tree(branching=3, depth=4, shortcut_density=0.50),
        shortcut_tree(branching=3, depth=4, shortcut_density=1.00),
        *room_graph_family(size=3),
        *behavioral_navigation_family(),
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
