"""Generate SVG galleries for the GeoMem task environments."""

from __future__ import annotations

from collections import defaultdict
from math import cos, pi, sin
from pathlib import Path
from typing import Callable

from commutativity_probe import (
    Environment,
    bottleneck_rooms,
    clamped_lattice,
    shortcut_tree,
    torus_lattice,
    trace_monoid,
    tree,
)
from plot_current_results import esc
from rnn_navigation_benchmark import (
    aliased_loop_rooms,
    aliased_rooms,
    aliased_t_maze,
    landmark_gap_detour,
    shortcut_route_choice,
)
from systematic_navigation_sweep import room_graph_environment, topology_edges


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
State = tuple[int, ...]
Point = tuple[float, float]

PANEL_W = 430
PANEL_H = 330
PAD = 28
NODE = "#14532d"
ROOT_NODE = "#be123c"
EDGE = "#52606d"
WRAP = "#0ea5e9"
SHORTCUT = "#d97706"
ROOM = "#e0f2fe"
TRACE = "#ede9fe"


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}"
        ".title{font-size:32px;font-weight:700}.subtitle{font-size:20px}"
        ".panel{font-size:23px;font-weight:700}.note{font-size:17px;fill:#52606d}"
        ".state{font-size:14px;fill:#334e68}.room{font-size:17px;font-weight:700}"
        "</style>",
        f'<text class="title" x="36" y="42">{esc(title)}</text>',
        f'<text class="subtitle" x="36" y="72">{esc(subtitle)}</text>',
    ]


def panel_frame(parts: list[str], x: float, y: float, title: str, note: str) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{PANEL_W}" height="{PANEL_H}" fill="#fbfdff" stroke="#d9e2ec" rx="6"/>')
    parts.append(f'<text class="panel" x="{x + 18}" y="{y + 31}">{esc(title)}</text>')
    parts.append(f'<text class="note" x="{x + 18}" y="{y + 56}">{esc(note)}</text>')


def valid_edges(env: Environment) -> list[tuple[State, State, str]]:
    edges: list[tuple[State, State, str]] = []
    seen: set[tuple[State, State, str]] = set()
    for state in env.states:
        for action in env.actions:
            if not env.is_valid(state, action):
                continue
            target = env.step(state, action)
            if target == state:
                continue
            pair = (state, target, action)
            reverse = (target, state, action)
            if pair in seen or reverse in seen:
                continue
            seen.add(pair)
            edges.append(pair)
    return edges


def scale_positions(positions: dict[State, Point], x: float, y: float) -> dict[State, Point]:
    left = x + PAD
    right = x + PANEL_W - PAD
    top = y + 82
    bottom = y + PANEL_H - PAD
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    span_x = max(1.0, max(xs) - min(xs))
    span_y = max(1.0, max(ys) - min(ys))
    return {
        state: (
            left + (point[0] - min(xs)) / span_x * (right - left),
            top + (point[1] - min(ys)) / span_y * (bottom - top),
        )
        for state, point in positions.items()
    }


def path_layout(states: tuple[State, ...], *, x_step: float = 1.0, row_gap: float = 1.0) -> dict[State, Point]:
    positions: dict[State, Point] = {(): (0.0, row_gap)}
    for state in states:
        if not state:
            continue
        positions[state] = (float(state[1] + 1) * x_step, float(state[0]) * row_gap)
    return positions


def hierarchy_layout(states: tuple[State, ...]) -> dict[State, Point]:
    levels: dict[int, list[State]] = defaultdict(list)
    for state in states:
        levels[len(state)].append(state)
    positions: dict[State, Point] = {}
    max_width = max(len(level) for level in levels.values())
    for depth, level in levels.items():
        ordered = sorted(level)
        for idx, state in enumerate(ordered):
            width = max(1, len(ordered) - 1)
            positions[state] = ((idx / width) * max_width, float(depth))
    return positions


def grid_layout(states: tuple[State, ...]) -> dict[State, Point]:
    if len(states[0]) == 2:
        return {state: (float(state[0]), -float(state[1])) for state in states}
    room_ids = sorted({state[0] for state in states})
    room_span = max(state[1] for state in states) + 1
    return {
        state: (float(room_ids.index(state[0]) * (room_span + 1) + state[1]), -float(state[2]))
        for state in states
    }


def circular_room_layout(states: tuple[State, ...]) -> dict[State, Point]:
    rooms = sorted({state[0] for state in states})
    size = max(state[1] for state in states) + 1
    positions: dict[State, Point] = {}
    for idx, room in enumerate(rooms):
        angle = -pi / 2 + 2 * pi * idx / len(rooms)
        cx = cos(angle) * 3.0
        cy = sin(angle) * 2.1
        for state in states:
            if state[0] == room:
                positions[state] = (cx + (state[1] - (size - 1) / 2) * 0.52, cy - (state[2] - (size - 1) / 2) * 0.52)
    return positions


def state_label(env: Environment, state: State) -> str:
    if env.name.startswith("trace_"):
        if len(state) > 2:
            return ""
        return "".join("ABC"[idx] for idx in state) or "root"
    if state == ():
        return "root"
    return ""


def edge_kind(env: Environment, state: State, target: State, action: str) -> str:
    if action in {"L", "R"} and env.name.startswith("shortcut_tree"):
        return "shortcut"
    if env.name == "torus_lattice" and len(state) == 2:
        if abs(state[0] - target[0]) > 1 or abs(state[1] - target[1]) > 1:
            return "wrap"
    return "normal"


def render_graph_panel(
    parts: list[str],
    env: Environment,
    x: float,
    y: float,
    title: str,
    note: str,
    layout: Callable[[tuple[State, ...]], dict[State, Point]],
    *,
    root: State | None = None,
    label_states: bool = False,
) -> None:
    panel_frame(parts, x, y, title, note)
    positions = scale_positions(layout(env.states), x, y)
    for state, target, action in valid_edges(env):
        if state not in positions or target not in positions:
            continue
        x1, y1 = positions[state]
        x2, y2 = positions[target]
        kind = edge_kind(env, state, target, action)
        color = EDGE
        width = 1.6
        dash = ""
        if kind == "wrap":
            color = WRAP
            dash = ' stroke-dasharray="5 4"'
        if kind == "shortcut":
            color = SHORTCUT
            width = 1.2
            dash = ' stroke-dasharray="4 3"'
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" opacity="0.72"{dash}/>')
    radius = 4.6 if len(env.states) < 55 else 3.0
    for state, (xx, yy) in positions.items():
        fill = ROOT_NODE if state == root or (root is None and state == ()) else NODE
        if env.name.startswith("trace_"):
            fill = ROOT_NODE if state == () else TRACE
            stroke = "#6d28d9"
        else:
            stroke = "#ffffff"
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>')
        label = state_label(env, state) if label_states else ""
        if label:
            parts.append(f'<text class="state" text-anchor="middle" x="{xx:.1f}" y="{yy - 8:.1f}">{esc(label)}</text>')


def add_grid_room_hints(parts: list[str], positions: dict[State, Point]) -> None:
    rooms: dict[int, list[Point]] = defaultdict(list)
    for state, point in positions.items():
        if len(state) == 3:
            rooms[state[0]].append(point)
    for room, points in rooms.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        parts.append(f'<rect x="{min(xs) - 10:.1f}" y="{min(ys) - 10:.1f}" width="{max(xs) - min(xs) + 20:.1f}" height="{max(ys) - min(ys) + 20:.1f}" fill="{ROOM}" opacity="0.35" stroke="#7cc4df" rx="4"/>')
        parts.append(f'<text class="room" x="{min(xs) - 2:.1f}" y="{min(ys) - 16:.1f}">room {room}</text>')


def render_room_panel(
    parts: list[str],
    env: Environment,
    x: float,
    y: float,
    title: str,
    note: str,
    layout: Callable[[tuple[State, ...]], dict[State, Point]],
    *,
    root: State | None = None,
) -> None:
    panel_frame(parts, x, y, title, note)
    positions = scale_positions(layout(env.states), x, y)
    add_grid_room_hints(parts, positions)
    for state, target, action in valid_edges(env):
        x1, y1 = positions[state]
        x2, y2 = positions[target]
        door = state[0] != target[0]
        color = "#0369a1" if door else EDGE
        width = 2.8 if door else 1.3
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" opacity="0.74"/>')
    for state, (xx, yy) in positions.items():
        fill = ROOT_NODE if state == root else NODE
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.1" fill="{fill}" stroke="#ffffff" stroke-width="1"/>')


def room_topology_layout(topology: str, states: tuple[State, ...]) -> dict[State, Point]:
    centers = {
        "loop_rich": {0: (-2.2, 0.0), 1: (0.0, -1.6), 2: (2.2, 0.0), 3: (0.0, 1.6)},
        "bottleneck": {0: (-3.0, -1.1), 1: (-3.0, 1.1), 2: (-1.0, 0.0), 3: (1.0, 0.0), 4: (3.0, -1.1), 5: (3.0, 1.1)},
        "tree_like": {0: (0.0, -2.0), 1: (-1.7, -0.4), 2: (1.7, -0.4), 3: (-2.8, 1.5), 4: (-0.8, 1.5), 5: (0.8, 1.5), 6: (2.8, 1.5)},
    }[topology]
    positions = {}
    for state in states:
        cx, cy = centers[state[0]]
        positions[state] = (cx + (state[1] - 1) * 0.34, cy - (state[2] - 1) * 0.34)
    return positions


def formal_gallery() -> str:
    panels = [
        (torus_lattice(4, 4), "Torus lattice", "Periodic local metric chart", grid_layout, (0, 0), False),
        (clamped_lattice(4, 4), "Clamped lattice", "Same grid; finite boundaries matter", grid_layout, (0, 0), False),
        (bottleneck_rooms(3), "Bottleneck rooms", "Locally flat rooms, one global door", grid_layout, (0, 0, 0), False),
        (shortcut_tree(branching=2, depth=4, shortcut_density=1.0), "Shortcut tree", "Tree order plus lateral shortcuts", hierarchy_layout, (), False),
        (trace_monoid(frozenset(), max_depth=3), "Trace rho 0", "No swaps: action order is state", hierarchy_layout, (), True),
        (trace_monoid(frozenset({("A", "B")}), max_depth=3), "Trace rho 0.33", "One permitted adjacent swap", hierarchy_layout, (), True),
        (trace_monoid(frozenset({("A", "B"), ("A", "C")}), max_depth=3), "Trace rho 0.67", "Two commuting action pairs", hierarchy_layout, (), True),
        (trace_monoid(frozenset({("A", "B"), ("A", "C"), ("B", "C")}), max_depth=3), "Trace rho 1", "Counts quotient all action orders", hierarchy_layout, (), True),
    ]
    width = 4 * PANEL_W + 5 * 22
    height = 122 + 2 * PANEL_H + 3 * 22
    parts = svg_start(width, height, "Formal environment gallery", "Prototypes used to separate local metric structure, tree order, shortcuts, and trace quotients.")
    for idx, (env, title, note, layout, root, labels) in enumerate(panels):
        col, row = idx % 4, idx // 4
        x = 22 + col * (PANEL_W + 22)
        y = 110 + row * (PANEL_H + 22)
        if env.name == "bottleneck_rooms":
            render_room_panel(parts, env, x, y, title, note, layout, root=root)
        else:
            render_graph_panel(parts, env, x, y, title, note, layout, root=root, label_states=labels)
    parts.append("</svg>")
    return "\n".join(parts)


def navigation_gallery() -> str:
    panels = [
        (aliased_rooms(3), "Aliased rooms", "Room identity hidden in sparse input", grid_layout, (0, 0, 0), "room"),
        (aliased_loop_rooms(2, 3), "Aliased loop rooms", "Repeated local rooms close a loop", circular_room_layout, (0, 0, 0), "room"),
        (aliased_t_maze(8), "Aliased T-maze", "Cue, blank corridor, final choice", path_layout, (), "path"),
        (landmark_gap_detour(), "Landmark gap detour", "Route history spans the sensory gap", path_layout, (), "path"),
        (shortcut_route_choice(), "Shortcut route choice", "Short and turning routes meet choice", path_layout, (), "path"),
        (tree(branching=3, depth=3), "Tree navigation", "Formal branch identity demands order", hierarchy_layout, (), "graph"),
    ]
    width = 3 * PANEL_W + 4 * 22
    height = 122 + 2 * PANEL_H + 3 * 22
    parts = svg_start(width, height, "Named navigation task gallery", "Latent transition graphs for the behavior-facing supervised benchmarks.")
    for idx, (env, title, note, layout, root, kind) in enumerate(panels):
        col, row = idx % 3, idx // 3
        x = 22 + col * (PANEL_W + 22)
        y = 110 + row * (PANEL_H + 22)
        if kind == "room":
            render_room_panel(parts, env, x, y, title, note, layout, root=root)
        else:
            render_graph_panel(parts, env, x, y, title, note, layout, root=root)
    parts.append("</svg>")
    return "\n".join(parts)


def room_graph_gallery() -> str:
    width = 3 * PANEL_W + 4 * 22
    height = 122 + PANEL_H + 2 * 22
    parts = svg_start(width, height, "Factorized room-graph environments", "Each local room is metric; global room topology varies while aliasing and cue visibility are crossed separately.")
    notes = {
        "loop_rich": "Loops and redundant inter-room routes",
        "bottleneck": "Two room clusters share one bridge",
        "tree_like": "No global cycles; branch order matters",
    }
    for idx, topology in enumerate(("loop_rich", "bottleneck", "tree_like")):
        env = room_graph_environment(topology)
        _, edges = topology_edges(topology)
        title = topology.replace("_", " ").title()
        note = f"{notes[topology]}; {len(edges)} doors"
        render_room_panel(
            parts,
            env,
            22 + idx * (PANEL_W + 22),
            110,
            title,
            note,
            lambda states, topology=topology: room_topology_layout(topology, states),
            root=(0, 1, 1),
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(name: str, content: str) -> None:
    (FIGURES / name).write_text(content, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_svg("formal_environment_gallery.svg", formal_gallery())
    write_svg("navigation_environment_gallery.svg", navigation_gallery())
    write_svg("room_graph_environment_gallery.svg", room_graph_gallery())
    print("Generated:")
    print((FIGURES / "formal_environment_gallery.svg").relative_to(ROOT))
    print((FIGURES / "navigation_environment_gallery.svg").relative_to(ROOT))
    print((FIGURES / "room_graph_environment_gallery.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
