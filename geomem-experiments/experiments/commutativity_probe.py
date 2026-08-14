"""Compatibility wrapper for environment diagnostics now owned by geomem_env."""

from __future__ import annotations

import sys
from pathlib import Path

ENV_SRC = Path(__file__).resolve().parents[2] / "geomem-env" / "src"
if str(ENV_SRC) not in sys.path:
    sys.path.insert(0, str(ENV_SRC))

from geomem_env import (
    Action,
    Environment,
    State,
    Transition,
    action_strings,
    aliased_loop_rooms,
    aliased_rooms,
    aliased_t_maze,
    behavioral_navigation_family,
    bottleneck_rooms,
    clamped_lattice,
    default_environments,
    door_state,
    effective_commutativity,
    gaussian_sensory_channel,
    graph_edges,
    landmark_gap_detour,
    lattice,
    pairwise_commutator_matrix,
    room_graph_environment,
    room_graph_family,
    sensory_alias_key,
    shortcut_route_choice,
    shortcut_tree,
    swapped_pairs,
    topology_edges,
    torus_lattice,
    trace_env,
    trace_monoid,
    trace_monoid_family,
    trace_monoid_general,
    tree,
)

__all__ = [
    "Action",
    "Environment",
    "State",
    "Transition",
    "action_strings",
    "aliased_loop_rooms",
    "aliased_rooms",
    "aliased_t_maze",
    "behavioral_navigation_family",
    "bottleneck_rooms",
    "clamped_lattice",
    "default_environments",
    "door_state",
    "effective_commutativity",
    "gaussian_sensory_channel",
    "graph_edges",
    "landmark_gap_detour",
    "lattice",
    "pairwise_commutator_matrix",
    "room_graph_environment",
    "room_graph_family",
    "sensory_alias_key",
    "shortcut_route_choice",
    "shortcut_tree",
    "swapped_pairs",
    "topology_edges",
    "torus_lattice",
    "trace_env",
    "trace_monoid",
    "trace_monoid_family",
    "trace_monoid_general",
    "tree",
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
