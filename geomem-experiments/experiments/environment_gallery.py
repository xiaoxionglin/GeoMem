"""Generate SVG galleries for the GeoMem task environments.

The environment definitions live in ``geomem_env``. This script only owns
paper-facing and atlas-style SVG rendering.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
from math import cos, pi, sin
from pathlib import Path
import sys
from typing import Callable

ENV_SRC = Path(__file__).resolve().parents[2] / "geomem-env" / "src"
if str(ENV_SRC) not in sys.path:
    sys.path.insert(0, str(ENV_SRC))

from geomem_env import (
    Environment,
    aliased_loop_rooms,
    aliased_rooms,
    aliased_t_maze,
    bottleneck_rooms,
    clamped_lattice,
    default_environments,
    environment_summary,
    landmark_gap_detour,
    residual_history_diagnostics,
    room_graph_environment,
    shortcut_route_choice,
    shortcut_tree,
    standard_observation_specs,
    topology_edges,
    torus_lattice,
    trace_monoid,
    tree,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
State = tuple[int, ...]
Point = tuple[float, float]

PANEL_W = 430
PANEL_H = 330
ATLAS_W = 390
ATLAS_H = 292
PAD = 28

NODE = "#14532d"
ROOT_NODE = "#be123c"
EDGE = "#52606d"
WRAP = "#0ea5e9"
SHORTCUT = "#d97706"
DOOR = "#0369a1"
ROOM = "#e0f2fe"
TRACE = "#ede9fe"
TEXT = "#1f2933"
MUTED = "#52606d"


def esc(value: object) -> str:
    return escape(str(value), quote=True)


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.2f}"


def residual_level(env: Environment) -> float:
    scores = []
    for spec in standard_observation_specs(env):
        if spec.name == "full":
            continue
        score = residual_history_diagnostics(env, spec.observe)["residual_history_score"]
        if score == score:
            scores.append(score)
    return max(scores) if scores else 0.0


def residual_color(value: float) -> str:
    value = max(0.0, min(1.0, value if value == value else 0.0))
    low = (220, 252, 231)
    mid = (253, 224, 71)
    high = (239, 68, 68)
    if value <= 0.5:
        t = value / 0.5
        a, b = low, mid
    else:
        t = (value - 0.5) / 0.5
        a, b = mid, high
    r = round(a[0] + (b[0] - a[0]) * t)
    g = round(a[1] + (b[1] - a[1]) * t)
    b_ = round(a[2] + (b[2] - a[2]) * t)
    return f"rgb({r},{g},{b_})"


def text_lines(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>"
        f"text{{font-family:Arial,Helvetica,sans-serif;fill:{TEXT}}}"
        ".title{font-size:32px;font-weight:700}.subtitle{font-size:19px;fill:#52606d}"
        ".panel{font-size:21px;font-weight:700}.note{font-size:15px;fill:#52606d}"
        ".metric{font-size:13px;fill:#334e68}.state{font-size:12px;fill:#334e68}.room{font-size:15px;font-weight:700}"
        ".badge{font-size:12px;font-weight:700}.legend{font-size:13px;fill:#52606d}"
        "</style>",
        f'<text class="title" x="36" y="42">{esc(title)}</text>',
        f'<text class="subtitle" x="36" y="70">{esc(subtitle)}</text>',
    ]


def panel_frame(parts: list[str], x: float, y: float, width: float, height: float, env: Environment, title: str, note: str) -> None:
    summary = environment_summary(env)
    residual = residual_level(env)
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#fbfdff" stroke="#d9e2ec" rx="6"/>')
    parts.append(f'<text class="panel" x="{x + 16}" y="{y + 28}">{esc(title)}</text>')
    parts.append(f'<rect x="{x + width - 93}" y="{y + 13}" width="72" height="22" fill="{residual_color(residual)}" stroke="#ffffff" rx="4"/>')
    parts.append(f'<text class="badge" text-anchor="middle" x="{x + width - 57}" y="{y + 29}">R {fmt(residual)}</text>')
    for line_idx, line in enumerate(text_lines(note, 44)[:2]):
        parts.append(f'<text class="note" x="{x + 16}" y="{y + 51 + line_idx * 17}">{esc(line)}</text>')
    metric = f"{summary['family']} / {summary['geometry']} | S={summary['n_states']} A={summary['n_actions']} V={summary['n_valid_transitions']}"
    parts.append(f'<text class="metric" x="{x + 16}" y="{y + 84}">{esc(metric)}</text>')


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


def scale_positions(positions: dict[State, Point], x: float, y: float, width: float, height: float) -> dict[State, Point]:
    left = x + PAD
    right = x + width - PAD
    top = y + 108
    bottom = y + height - PAD
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


def path_layout(states: tuple[State, ...]) -> dict[State, Point]:
    positions: dict[State, Point] = {(): (0.0, 1.0)}
    for state in states:
        if not state:
            continue
        positions[state] = (float(state[1] + 1), float(state[0]))
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


def state_label(env: Environment, state: State) -> str:
    if env.name.startswith("trace_"):
        if len(state) > 2:
            return ""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "".join(alphabet[idx] for idx in state) or "root"
    if state == ():
        return "root"
    return ""


def edge_kind(env: Environment, state: State, target: State, action: str) -> str:
    if action in {"L", "R"} and env.name.startswith("shortcut_tree"):
        return "shortcut"
    if len(state) == 3 and len(target) == 3 and state[0] != target[0]:
        return "door"
    if env.name == "torus_lattice" and len(state) == 2:
        if abs(state[0] - target[0]) > 1 or abs(state[1] - target[1]) > 1:
            return "wrap"
    return "normal"


def render_edges_and_nodes(
    parts: list[str],
    env: Environment,
    positions: dict[State, Point],
    *,
    root: State | None,
    label_states: bool,
    max_edges: int = 700,
) -> None:
    for state, target, action in valid_edges(env)[:max_edges]:
        if state not in positions or target not in positions:
            continue
        x1, y1 = positions[state]
        x2, y2 = positions[target]
        kind = edge_kind(env, state, target, action)
        color = EDGE
        width = 1.3
        dash = ""
        if kind == "wrap":
            color = WRAP
            dash = ' stroke-dasharray="5 4"'
        elif kind == "shortcut":
            color = SHORTCUT
            width = 1.1
            dash = ' stroke-dasharray="4 3"'
        elif kind == "door":
            color = DOOR
            width = 2.6
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" opacity="0.70"{dash}/>')

    radius = 4.3 if len(positions) < 55 else 2.9
    for state, (xx, yy) in positions.items():
        fill = ROOT_NODE if state == root or (root is None and state == ()) else NODE
        stroke = "#ffffff"
        if env.name.startswith("trace_"):
            fill = ROOT_NODE if state == () else TRACE
            stroke = "#6d28d9"
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
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
        parts.append(f'<rect x="{min(xs) - 8:.1f}" y="{min(ys) - 8:.1f}" width="{max(xs) - min(xs) + 16:.1f}" height="{max(ys) - min(ys) + 16:.1f}" fill="{ROOM}" opacity="0.33" stroke="#7cc4df" rx="4"/>')
        parts.append(f'<text class="room" x="{min(xs) - 2:.1f}" y="{min(ys) - 13:.1f}">room {room}</text>')


def render_graph_panel(
    parts: list[str],
    env: Environment,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    note: str,
    layout: Callable[[tuple[State, ...]], dict[State, Point]],
    *,
    root: State | None = None,
    label_states: bool = False,
    room_hints: bool = False,
) -> None:
    panel_frame(parts, x, y, width, height, env, title, note)
    positions = scale_positions(layout(env.states), x, y, width, height)
    if room_hints:
        add_grid_room_hints(parts, positions)
    render_edges_and_nodes(parts, env, positions, root=root, label_states=label_states)


def render_trace_abstract_panel(parts: list[str], env: Environment, x: float, y: float, width: float, height: float, title: str, note: str) -> None:
    panel_frame(parts, x, y, width, height, env, title, note)
    levels: dict[int, int] = defaultdict(int)
    for state in env.states:
        levels[len(state)] += 1
    max_count = max(levels.values())
    left = x + 38
    right = x + width - 38
    top = y + 116
    bottom = y + height - 42
    max_depth = max(levels)
    for depth in range(max_depth + 1):
        yy = top + (depth / max(1, max_depth)) * (bottom - top)
        count = levels[depth]
        bar_w = max(8.0, (count / max_count) * (right - left))
        parts.append(f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{left + bar_w:.1f}" y2="{yy:.1f}" stroke="#7c3aed" stroke-width="7" opacity="0.70"/>')
        parts.append(f'<text class="metric" x="{left:.1f}" y="{yy - 8:.1f}">d{depth}: {count}</text>')
    parts.append(f'<text class="legend" x="{left:.1f}" y="{bottom + 27:.1f}">Depth bands show canonical trace states, not every transition.</text>')


def panel_for_env(parts: list[str], env: Environment, x: float, y: float, width: float, height: float, title: str | None = None, note: str | None = None) -> None:
    title = title or env.name.replace("_", " ")
    summary = environment_summary(env)
    note = note or f"{summary['family']} / {summary['geometry']}"
    family = summary["family"]
    if family == "trace_monoid" and len(env.states) > 220:
        render_trace_abstract_panel(parts, env, x, y, width, height, title, note)
        return
    if family in {"rooms", "room_graph", "aliased_navigation"} and all(len(state) == 3 for state in env.states):
        if env.name.startswith("room_graph_"):
            topology = env.name.replace("room_graph_", "")
            layout = lambda states, topology=topology: room_topology_layout(topology, states)
        elif env.name == "aliased_loop_rooms":
            layout = circular_room_layout
        else:
            layout = grid_layout
        render_graph_panel(parts, env, x, y, width, height, title, note, layout, root=env.start_states[0] if env.start_states else None, room_hints=True)
        return
    if env.name.startswith("trace_"):
        render_graph_panel(parts, env, x, y, width, height, title, note, hierarchy_layout, root=(), label_states=True)
        return
    if family in {"tree", "shortcut_tree"}:
        render_graph_panel(parts, env, x, y, width, height, title, note, hierarchy_layout, root=())
        return
    if family in {"cue_gap", "route_choice"}:
        render_graph_panel(parts, env, x, y, width, height, title, note, path_layout, root=())
        return
    render_graph_panel(parts, env, x, y, width, height, title, note, grid_layout, root=env.start_states[0] if env.start_states else None)


def add_legend(parts: list[str], x: float, y: float) -> None:
    items = [
        (EDGE, "valid edge"),
        (DOOR, "door/global transition"),
        (WRAP, "wrap edge"),
        (SHORTCUT, "shortcut"),
        (ROOT_NODE, "start/root"),
    ]
    for idx, (color, label) in enumerate(items):
        xx = x + idx * 156
        parts.append(f'<line x1="{xx}" y1="{y}" x2="{xx + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text class="legend" x="{xx + 36}" y="{y + 4}">{esc(label)}</text>')
    parts.append(f'<text class="legend" x="{x}" y="{y + 30}">R badge = max residual history demand across non-full standard observations.</text>')


def formal_gallery() -> str:
    panels = [
        (torus_lattice(4, 4), "Torus lattice", "Periodic metric control; actions commute."),
        (clamped_lattice(4, 4), "Clamped lattice", "Boundaries create raw finite-task effects."),
        (bottleneck_rooms(3), "Bottleneck rooms", "Locally metric, one global door."),
        (shortcut_tree(branching=2, depth=4, shortcut_density=1.0), "Shortcut tree", "Branch order plus lateral edges."),
        (trace_monoid(frozenset(), max_depth=3), "Trace rho 0", "No swaps: order is state."),
        (trace_monoid(frozenset({("A", "B")}), max_depth=3), "Trace rho 0.33", "One commuting action pair."),
        (trace_monoid(frozenset({("A", "B"), ("A", "C")}), max_depth=3), "Trace rho 0.67", "Two commuting action pairs."),
        (trace_monoid(frozenset({("A", "B"), ("A", "C"), ("B", "C")}), max_depth=3), "Trace rho 1", "Counts quotient all orders."),
    ]
    width = 4 * PANEL_W + 5 * 22
    height = 138 + 2 * PANEL_H + 3 * 22
    parts = svg_start(width, height, "Formal environment gallery", "Representative controlled geometries with size and residual-history badges.")
    add_legend(parts, 36, 96)
    for idx, (env, title, note) in enumerate(panels):
        col, row = idx % 4, idx // 4
        panel_for_env(parts, env, 22 + col * (PANEL_W + 22), 126 + row * (PANEL_H + 22), PANEL_W, PANEL_H, title, note)
    parts.append("</svg>")
    return "\n".join(parts)


def navigation_gallery() -> str:
    panels = [
        (aliased_rooms(3), "Aliased rooms", "Room identity hidden in sparse observations."),
        (aliased_loop_rooms(2, 3), "Aliased loop rooms", "Repeated rooms close a loop."),
        (aliased_t_maze(8), "Aliased T-maze", "Cue, blank corridor, choice."),
        (landmark_gap_detour(), "Landmark gap detour", "Cue memory across a route gap."),
        (shortcut_route_choice(), "Shortcut route choice", "Route history selects final arm."),
        (tree(branching=3, depth=3), "Tree navigation", "Branch identity demands order."),
    ]
    width = 3 * PANEL_W + 4 * 22
    height = 138 + 2 * PANEL_H + 3 * 22
    parts = svg_start(width, height, "Named navigation task gallery", "Behavior-facing tasks with aliasing, cue gaps, and route-history demand.")
    add_legend(parts, 36, 96)
    for idx, (env, title, note) in enumerate(panels):
        col, row = idx % 3, idx // 3
        panel_for_env(parts, env, 22 + col * (PANEL_W + 22), 126 + row * (PANEL_H + 22), PANEL_W, PANEL_H, title, note)
    parts.append("</svg>")
    return "\n".join(parts)


def room_graph_gallery() -> str:
    width = 3 * PANEL_W + 4 * 22
    height = 138 + PANEL_H + 2 * 22
    parts = svg_start(width, height, "Factorized room-graph environments", "Local room geometry is fixed; global room topology varies.")
    add_legend(parts, 36, 96)
    notes = {
        "loop_rich": "Loops and redundant inter-room routes.",
        "bottleneck": "Two clusters share a bridge.",
        "tree_like": "No global cycles; branch order matters.",
    }
    for idx, topology in enumerate(("loop_rich", "bottleneck", "tree_like")):
        env = room_graph_environment(topology)
        _, edges = topology_edges(topology)
        panel_for_env(
            parts,
            env,
            22 + idx * (PANEL_W + 22),
            126,
            PANEL_W,
            PANEL_H,
            topology.replace("_", " ").title(),
            f"{notes[topology]} {len(edges)} doors.",
        )
    parts.append("</svg>")
    return "\n".join(parts)


def all_environment_atlas() -> str:
    envs = default_environments()
    cols = 4
    rows = (len(envs) + cols - 1) // cols
    width = cols * ATLAS_W + (cols + 1) * 18
    height = 150 + rows * ATLAS_H + (rows + 1) * 18
    parts = svg_start(width, height, "All environment atlas", "All default geomem-env environments grouped by metadata, with complexity and residual-history badges.")
    add_legend(parts, 36, 98)
    for idx, env in enumerate(envs):
        col, row = idx % cols, idx // cols
        x = 18 + col * (ATLAS_W + 18)
        y = 138 + row * (ATLAS_H + 18)
        panel_for_env(parts, env, x, y, ATLAS_W, ATLAS_H)
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg(name: str, content: str) -> None:
    (FIGURES / name).write_text(content, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "formal_environment_gallery.svg": formal_gallery(),
        "navigation_environment_gallery.svg": navigation_gallery(),
        "room_graph_environment_gallery.svg": room_graph_gallery(),
        "all_environment_atlas.svg": all_environment_atlas(),
    }
    for name, content in outputs.items():
        write_svg(name, content)
    print("Generated:")
    for name in outputs:
        print((FIGURES / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
