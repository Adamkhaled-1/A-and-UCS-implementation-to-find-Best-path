# -*- coding: utf-8 -*-
"""Warehouse_Delivery_Robot_Search_Algorithms.ipynb

# Autonomous Warehouse Delivery Robot: Discrete Pathfinding Search Algorithms

**Course:** Intelligent Systems / Artificial Intelligence  
**Topic:** Discrete Graph Search Algorithms (Uniform Cost Search vs. A* Search)  
**Repository:** [warehouse-delivery-robot-search-algorithms](https://github.com/shaheenmohammedsh-ops/warehouse-delivery-robot-search-algorithms)  

---

## 1. Project Overview & Problem Statement

In an automated warehouse, an autonomous mobile robot must deliver inventory from a start dock ($S$) to a destination goal ($G$) across a 2D grid floor with obstacles and varying surface friction. Because rough floor patches cost more to traverse than normal floor tiles, the shortest path in steps is not necessarily the cheapest route.

This project implements and evaluates two pathfinding algorithms from scratch:
1. **Uniform Cost Search (UCS):** An uninformed algorithm that expands states in order of accumulated path cost $g(n)$, guaranteeing cost-optimality for positive edge weights.
2. **A* Search:** An informed algorithm that guides exploration using the **Manhattan distance heuristic**, balancing accumulated path cost $g(n)$ with estimated remaining distance $h(n)$.

## 2. Problem Formulation and Cost Model

We formalize the warehouse navigation task as a discrete graph search problem:

| Element | Description |
| :--- | :--- |
| **State Space ($\mathcal{S}$)** | Grid coordinates $(r, c)$ where $0 \le r < R$ and $0 \le c < C$. |
| **Start State ($S$)** | Initial dock location $(r_S, c_S)$, located dynamically from the grid. |
| **Goal State ($G$)** | Target delivery location $(r_G, c_G)$, located dynamically from the grid. |
| **Action Space ($\mathcal{A}$)** | 4 orthogonal moves: Up $(-1, 0)$, Down $(+1, 0)$, Left $(0, -1)$, and Right $(0, +1)$. Diagonal moves are disallowed. |
| **Transition Model** | Moving from $(r, c)$ to an adjacent in-bounds, obstacle-free cell $(r', c')$. |
| **Step Cost $c(s, s')$** | Determined strictly by the tile type of the **entering cell** $s'$: |
| | $\bullet$ **Normal floor (`.`):** cost = **1** |
| | $\bullet$ **Rough floor (`~`):** cost = **3** (high friction / debris) |
| | $\bullet$ **Goal dock (`G`):** cost = **1** (entering destination) |
| | $\bullet$ **Start dock (`S`):** cost = **0** (origin point, not counted) |
| | $\bullet$ **Obstacle (`#`):** impassable barrier (infinite cost) |

### Evaluation Metrics
Each search run records five standardized metrics: **Path Length** (moves from $S$ to $G$), **Path Cost** (sum of entered tile costs), **Nodes Expanded** (states popped and processed), **Maximum Frontier Size** (peak queue memory), and **Execution Time** (wall-clock milliseconds via `time.perf_counter()`).

## 3. Environment Setup, Map Definitions, and Configurable `GRID`

We define the three benchmark maps below:
- `MAP_A`: Standard warehouse floor with structural wall pillars and rough patches.
- `MAP_B`: Designed to evaluate the cost-vs-step tradeoff between a short rough path and a longer clean detour.
- `MAP_C`: Contains an unbroken wall barrier that makes the goal unreachable.

> **Note on `GRID` Configuration:** The `GRID` variable sets the active map for the single-map interactive demonstration in Section 9.1. Sections 5 through 10 run the complete three-map benchmark suite independently.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Table display configuration
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Clean plot styling defaults
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.edgecolor': '#CED4DA',
    'axes.linewidth': 1.0,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF'
})


# BENCHMARK MAP DEFINITIONS

MAP_A: List[str] = [
    "S..#....",
    "##.#.##.",
    "....~~..",
    ".####.#.",
    "...~..#G",
]

MAP_B: List[str] = [
    "S~~~~~~~G",
    ".#######.",
    ".........",
    ".#.#.#.#.",
    ".........",
]

MAP_C: List[str] = [
    "S..#....",
    ".#.#.##.",
    "...#..#.",
    ".###.##.",
    "....#..G",
]

# Active map selector for single-map demonstration (default: MAP_A)
GRID: List[str] = MAP_A

# Movement cost lookup based strictly on entering tile type
COST_TABLE: Dict[str, int] = {
    '.': 1,   # Normal floor
    '~': 3,   # Rough floor (elevated friction)
    'G': 1,   # Goal docking tile
    'S': 0,   # Start dock (origin, not counted in traversal)
}

WALL_CHAR: str = '#'
VALID_CELL_CHARS: Set[str] = {'.', '~', 'G', 'S', WALL_CHAR}

# 4-connected cardinal movements: (delta_row, delta_col)
DIRECTIONS: List[Tuple[int, int]] = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1),   # Right
]

Coordinate = Tuple[int, int]

"""## 4. Shared Data Structures and Validation
- **Grid Parsing & Validation:** Dynamically locates `S` and `G` and verifies grid rectangularity.
- **Neighbor Generation:** Produces valid in-bounds, obstacle-free 4-connected transitions.
- **Path Reconstruction:** Backtracks pointers from `G` to `S` using the parent map.
- **Solution Verification:** Automated assertions that confirm path continuity, validity, and cost accuracy.
"""

# DATA STRUCTURES
@dataclass(frozen=True)
class GridInfo:
    """Encapsulates a validated warehouse grid environment."""
    grid: List[str]
    rows: int
    cols: int
    start: Coordinate
    goal: Coordinate


@dataclass
class SearchResult:
    """Standardized container for search execution metrics."""
    algorithm: str
    map_name: str
    found: bool
    path: List[Coordinate] = field(default_factory=list)
    length: int = 0
    cost: float = float('inf')
    nodes_expanded: int = 0
    max_frontier_size: int = 0
    execution_time_ms: float = 0.0

    @property
    def status(self) -> str:
        return 'Success' if self.found else 'No path found'


def parse_and_validate_grid(grid: List[str]) -> GridInfo:
    """Validates grid structure and dynamically scans for start 'S' and goal 'G'."""
    if not grid or not isinstance(grid, list):
        raise ValueError('Grid must be a non-empty list of strings.')

    rows = len(grid)
    cols = len(grid[0])
    if cols == 0:
        raise ValueError('Grid rows cannot be empty.')

    start_pos: Optional[Coordinate] = None
    goal_pos: Optional[Coordinate] = None

    # Scan grid dynamically to find unique start and goal coordinates
    for r, row_str in enumerate(grid):
        if len(row_str) != cols:
            raise ValueError(f'Inconsistent column count at row {r}: expected {cols}, found {len(row_str)}.')
        for c, char in enumerate(row_str):
            if char not in VALID_CELL_CHARS:
                raise ValueError(f"Invalid character '{char}' at coordinate ({r}, {c}).")
            if char == 'S':
                if start_pos is not None:
                    raise ValueError(f'Duplicate start position S detected at ({r}, {c}).')
                start_pos = (r, c)
            elif char == 'G':
                if goal_pos is not None:
                    raise ValueError(f'Duplicate goal position G detected at ({r}, {c}).')
                goal_pos = (r, c)

    if start_pos is None:
        raise ValueError("Missing start position 'S' in grid.")
    if goal_pos is None:
        raise ValueError("Missing goal position 'G' in grid.")

    return GridInfo(grid=grid, rows=rows, cols=cols, start=start_pos, goal=goal_pos)


def get_valid_neighbors(pos: Coordinate, grid_info: GridInfo) -> List[Tuple[Coordinate, int]]:
    """Generates valid 4-connected in-bounds neighbors and their entering step costs."""
    r, c = pos
    neighbors: List[Tuple[Coordinate, int]] = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        # Check grid boundaries
        if 0 <= nr < grid_info.rows and 0 <= nc < grid_info.cols:
            char = grid_info.grid[nr][nc]
            # Filter impassable obstacle walls
            if char != WALL_CHAR:
                cost = COST_TABLE[char]
                neighbors.append(((nr, nc), cost))
    return neighbors


def reconstruct_path(came_from: Dict[Coordinate, Optional[Coordinate]], goal: Coordinate) -> List[Coordinate]:
    """Backtracks parent pointers from Goal to Start to build the trajectory."""
    path: List[Coordinate] = []
    curr: Optional[Coordinate] = goal
    while curr is not None:
        path.append(curr)
        curr = came_from.get(curr)
    path.reverse()
    return path


def validate_path_solution(grid: List[str], result: SearchResult) -> bool:
    """Asserts path validity, boundary constraints, continuity, and cost correctness."""
    grid_info = parse_and_validate_grid(grid)
    if not result.found:
        assert len(result.path) == 0, 'Unsuccessful search must report empty path.'
        assert result.cost == float('inf'), 'Unsuccessful search must report infinite cost.'
        return True

    path = result.path
    assert len(path) >= 1, 'Successful path cannot be empty.'
    assert path[0] == grid_info.start, f'Path starts at {path[0]}, expected {grid_info.start}.'
    assert path[-1] == grid_info.goal, f'Path ends at {path[-1]}, expected {grid_info.goal}.'
    assert result.length == len(path) - 1, f'Path length ({result.length}) != moves ({len(path)-1}).'

    recalculated_cost = 0
    for i in range(1, len(path)):
        prev, curr = path[i - 1], path[i]
        step_dist = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1])
        assert step_dist == 1, f'Non-cardinal step from {prev} to {curr}.'
        r, c = curr
        char = grid_info.grid[r][c]
        assert char != WALL_CHAR, f'Path traverses obstacle at ({r}, {c}).'
        recalculated_cost += COST_TABLE[char]

    assert recalculated_cost == result.cost, (
        f'Cost mismatch: recomputed {recalculated_cost} != reported {result.cost}.'
    )
    return True

"""# **Visualization:** Renders ASCII overlays and publication-quality 2D grid plots.
---
"""

def render_ascii_path(grid: List[str], path: List[Coordinate]) -> str:
    """Renders ASCII grid with '*' along path while preserving 'S' and 'G'."""
    grid_info = parse_and_validate_grid(grid)
    path_set = set(path)
    rendered = []
    for r in range(grid_info.rows):
        row_chars = []
        for c in range(grid_info.cols):
            char = grid_info.grid[r][c]
            if char in ('S', 'G'):
                row_chars.append(char)
            elif (r, c) in path_set:
                row_chars.append('*')
            else:
                row_chars.append(char)
        rendered.append(''.join(row_chars))
    return '\n'.join(rendered)


# Color palette for 2D visualizer
COLOR_NORMAL = '#F8F9FA'       # Normal floor '.' (cost 1)
COLOR_ROUGH = '#E9C46A'        # Rough floor '~' (cost 3)
COLOR_WALL = '#264653'         # Obstacle '#' (impassable)
COLOR_START = '#2A9D8F'        # Start 'S' (cost 0)
COLOR_GOAL = '#E76F51'         # Goal 'G' (cost 1)
COLOR_PATH_LINE = '#1D3557'    # Trajectory path line
COLOR_PATH_MARKER = '#457B9D'  # Trajectory waypoint marker
COLOR_GRID_LINE = '#CED4DA'    # Grid cell borders


def plot_warehouse_grid(grid: List[str], result: SearchResult) -> plt.Figure:
    """Generates a graphical 2D plot of the warehouse grid and optimal trajectory."""
    grid_info = parse_and_validate_grid(grid)
    rows, cols = grid_info.rows, grid_info.cols
    path_nodes = set(result.path) if result.found else set()

    fig, ax = plt.subplots(figsize=(max(6.0, cols * 0.9), max(4.0, rows * 0.85)), dpi=150)

    # Draw individual grid cells with terrain coloring
    for r in range(rows):
        for c in range(cols):
            char = grid_info.grid[r][c]
            if (r, c) == grid_info.start:
                fill_color = COLOR_START
            elif (r, c) == grid_info.goal:
                fill_color = COLOR_GOAL
            elif char == '#':
                fill_color = COLOR_WALL
            elif char == '~':
                fill_color = COLOR_ROUGH
            else:
                fill_color = COLOR_NORMAL

            rect = patches.Rectangle(
                (c, rows - 1 - r), 1, 1,
                linewidth=1.0, edgecolor=COLOR_GRID_LINE, facecolor=fill_color, zorder=1
            )
            ax.add_patch(rect)

            center_x = c + 0.5
            center_y = rows - 1 - r + 0.5

            if (r, c) == grid_info.start:
                ax.text(center_x, center_y, 'S\n(0)', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white', zorder=3)
            elif (r, c) == grid_info.goal:
                ax.text(center_x, center_y, 'G\n(1)', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white', zorder=3)
            elif char == '#':
                ax.text(center_x, center_y, '#', ha='center', va='center',
                        fontsize=13, fontweight='bold', color='#A8DADC', zorder=3)
            elif char == '~':
                if (r, c) not in path_nodes:
                    ax.text(center_x, center_y, '~\n(3)', ha='center', va='center',
                            fontsize=9, fontweight='bold', color='#6C584C', zorder=3)
                else:
                    ax.text(c + 0.82, rows - 1 - r + 0.82, '3', ha='center', va='center',
                            fontsize=7, fontweight='bold', color='#6C584C', zorder=3)
            else:
                if (r, c) not in path_nodes:
                    ax.text(center_x, center_y, '·\n(1)', ha='center', va='center',
                            fontsize=8, color='#6C757D', zorder=3)
                else:
                    ax.text(c + 0.82, rows - 1 - r + 0.82, '1', ha='center', va='center',
                            fontsize=7, color='#6C757D', zorder=3)

    # Draw continuous trajectory line and numbered step markers
    if result.found and result.path:
        xs = [c + 0.5 for _, c in result.path]
        ys = [rows - 1 - r + 0.5 for r, _ in result.path]
        ax.plot(xs, ys, color=COLOR_PATH_LINE, linewidth=3.0, alpha=0.9, zorder=5)
        ax.scatter(xs[1:-1], ys[1:-1], color=COLOR_PATH_MARKER, s=110,
                   edgecolor='white', linewidth=1.2, zorder=6)

        for step_idx, (r, c) in enumerate(result.path[1:-1], start=1):
            ax.annotate(str(step_idx), (c + 0.5, rows - 1 - r + 0.5),
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=7)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks([x + 0.5 for x in range(cols)])
    ax.set_xticklabels([f'C{x}' for x in range(cols)], fontsize=8)
    ax.set_yticks([rows - 1 - y + 0.5 for y in range(rows)])
    ax.set_yticklabels([f'R{y}' for y in range(rows)], fontsize=8)
    ax.tick_params(axis='both', which='both', length=0)

    cost_str = f'{result.cost:.0f}' if result.found else 'N/A'
    len_str = f'{result.length} moves' if result.found else 'N/A'
    ax.set_title(
        f'{result.map_name} — {result.algorithm}\n'
        f'Status: {result.status} | Cost: {cost_str} | Length: {len_str} | '
        f'Expanded: {result.nodes_expanded} | Peak Frontier: {result.max_frontier_size}',
        fontsize=10, fontweight='bold', pad=10, color='#1D3557'
    )

    legend_elements = [
        patches.Patch(facecolor=COLOR_START, edgecolor='none', label="Start 'S' (0)"),
        patches.Patch(facecolor=COLOR_GOAL, edgecolor='none', label="Goal 'G' (1)"),
        patches.Patch(facecolor=COLOR_NORMAL, edgecolor=COLOR_GRID_LINE, label="Normal '.' (1)"),
        patches.Patch(facecolor=COLOR_ROUGH, edgecolor=COLOR_GRID_LINE, label="Rough '~' (3)"),
        patches.Patch(facecolor=COLOR_WALL, edgecolor='none', label="Obstacle '#'"),
    ]
    if result.found:
        legend_elements.append(
            plt.Line2D([0], [0], color=COLOR_PATH_LINE, lw=2.5, marker='o',
                       markerfacecolor=COLOR_PATH_MARKER, markeredgecolor='white', label='Optimal Trajectory')
        )

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=7.5, frameon=True, facecolor='#F8F9FA', edgecolor='#CED4DA')

    plt.tight_layout()
    return fig

"""## 5. Uniform Cost Search (UCS) — Uninformed Search

### 5.1 Algorithm Overview
Uniform Cost Search (UCS) expands nodes in strictly non-decreasing order of accumulated path cost $g(n)$ from the start state $S$.

- **Priority Queue:** The frontier stores active paths ordered by $g(n)$, with an insertion counter to break ties deterministically.
- **Cost Tracking:** The dictionary `cost_so_far[node]` tracks the minimal cost discovered to each cell, allowing stale queue entries to be discarded immediately when popped.
- **Optimality Guarantee:** Performing the goal test upon **popping** from the priority queue ensures that the first time $G$ is removed, its accumulated path cost is globally optimal (since all step costs $c \ge 1$).
"""

def uniform_cost_search(grid: List[str], map_name: str = 'Grid') -> SearchResult:
    """
    Executes Uniform Cost Search (UCS) from Start to Goal.
    Expands nodes in increasing order of accumulated path cost g(n).
    """
    grid_info = parse_and_validate_grid(grid)
    start_time = time.perf_counter()

    # Priority queue stores: (g_cost, sequence_id, current_coordinate)
    frontier: List[Tuple[int, int, Coordinate]] = []
    counter = 0
    heapq.heappush(frontier, (0, counter, grid_info.start))

    # Track parent pointers for backtracking and cheapest known cost to each node
    came_from: Dict[Coordinate, Optional[Coordinate]] = {grid_info.start: None}
    cost_so_far: Dict[Coordinate, int] = {grid_info.start: 0}

    nodes_expanded = 0
    max_frontier_size = len(frontier)

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_g, _, current_node = heapq.heappop(frontier)

        # Skip stale entries if a cheaper route to this cell was already expanded
        if current_g > cost_so_far[current_node]:
            continue

        nodes_expanded += 1

        # Goal check upon queue removal guarantees cost-optimality
        if current_node == grid_info.goal:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path = reconstruct_path(came_from, grid_info.goal)
            return SearchResult(
                algorithm='Uniform Cost Search (UCS)',
                map_name=map_name,
                found=True,
                path=path,
                length=len(path) - 1,
                cost=cost_so_far[grid_info.goal],
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier_size,
                execution_time_ms=elapsed_ms,
            )

        # Expand valid 4-connected neighbors
        for neighbor, move_cost in get_valid_neighbors(current_node, grid_info):
            new_cost = current_g + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current_node
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbor))
                max_frontier_size = max(max_frontier_size, len(frontier))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm='Uniform Cost Search (UCS)',
        map_name=map_name,
        found=False,
        path=[],
        length=0,
        cost=float('inf'),
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier_size,
        execution_time_ms=elapsed_ms,
    )

"""### 5.2 UCS Benchmark Results on MAP_A
---
"""

# Execute UCS on MAP_A and validate solution
res_ucs_a = uniform_cost_search(MAP_A, map_name='MAP_A')
validate_path_solution(MAP_A, res_ucs_a)

print('=== MAP_A: Uniform Cost Search (UCS) Results ===')
print(f'Status:             {res_ucs_a.status}')
print(f'Path Coordinates:   {res_ucs_a.path}')
print(f'Path Length (moves):{res_ucs_a.length}')
print(f'Path Cost:          {res_ucs_a.cost}')
print(f'Nodes Expanded:     {res_ucs_a.nodes_expanded}')
print(f'Max Frontier Size:  {res_ucs_a.max_frontier_size}')
print(f'Runtime (ms):       {res_ucs_a.execution_time_ms:.3f} ms')
print('\nASCII Path Representation:')
print(render_ascii_path(MAP_A, res_ucs_a.path))

fig_ucs_a = plot_warehouse_grid(MAP_A, res_ucs_a)
plt.show()

"""### 5.3 UCS Benchmark Results on MAP_B — Step Count vs. Traversal Cost Tradeoff
---
"""

# Execute UCS on MAP_B and validate solution
res_ucs_b = uniform_cost_search(MAP_B, map_name='MAP_B')
validate_path_solution(MAP_B, res_ucs_b)

print('=== MAP_B: Uniform Cost Search (UCS) Results ===')
print(f'Status:             {res_ucs_b.status}')
print(f'Path Coordinates:   {res_ucs_b.path}')
print(f'Path Length (moves):{res_ucs_b.length}')
print(f'Path Cost:          {res_ucs_b.cost}')
print(f'Nodes Expanded:     {res_ucs_b.nodes_expanded}')
print(f'Max Frontier Size:  {res_ucs_b.max_frontier_size}')
print(f'Runtime (ms):       {res_ucs_b.execution_time_ms:.3f} ms')
print('\nASCII Path Representation:')
print(render_ascii_path(MAP_B, res_ucs_b.path))

fig_ucs_b = plot_warehouse_grid(MAP_B, res_ucs_b)
plt.show()

"""### 5.4 UCS Benchmark Results on MAP_C — Impassable Barrier & Unsolvable Termination
---
"""

# Execute UCS on MAP_C and validate unsolvable termination
res_ucs_c = uniform_cost_search(MAP_C, map_name='MAP_C')
validate_path_solution(MAP_C, res_ucs_c)

ucs_cost_c_str = f'{res_ucs_c.cost:.0f}' if res_ucs_c.found else 'N/A'
ucs_len_c_str = f'{res_ucs_c.length} moves' if res_ucs_c.found else 'N/A'

print('=== MAP_C: Uniform Cost Search (UCS) Results ===')
print(f'Status:             {res_ucs_c.status}')
if not res_ucs_c.found:
    print('No path found')
print(f'Path Coordinates:   {res_ucs_c.path}')
print(f'Path Length:        {ucs_len_c_str}')
print(f'Path Cost:          {ucs_cost_c_str}')
print(f'Nodes Expanded:     {res_ucs_c.nodes_expanded}')
print(f'Max Frontier Size:  {res_ucs_c.max_frontier_size}')
print(f'Runtime (ms):       {res_ucs_c.execution_time_ms:.3f} ms')

fig_ucs_c = plot_warehouse_grid(MAP_C, res_ucs_c)
plt.show()

"""## 6. Uniform Cost Search (UCS) Summary Table & Interpretation

The table below aggregates the empirical results for Uniform Cost Search across all benchmark maps:
"""

# Aggregate UCS benchmark results into a summary DataFrame
ucs_summary_data = [
    {
        'Map': res.map_name,
        'Algorithm': res.algorithm,
        'Status': res.status,
        'Path Length (Moves)': str(res.length) if res.found else 'N/A',
        'Path Cost': f'{res.cost:.0f}' if res.found else 'N/A',
        'Nodes Expanded': res.nodes_expanded,
        'Max Frontier Size': res.max_frontier_size,
        'Runtime (ms)': f'{res.execution_time_ms:.3f}'
    }
    for res in [res_ucs_a, res_ucs_b, res_ucs_c]
]
df_ucs_summary = pd.DataFrame(ucs_summary_data)
df_ucs_summary

"""### Interpretation of UCS Performance
1. **Cost Optimality:** UCS successfully identifies the lowest-cost path on solvable layouts (`MAP_A` and `MAP_B`).
2. **Search Overhead:** Lacking directional guidance, UCS expands radially outward in all directions, exploring nearly the entire grid on `MAP_B` (31 nodes) before reaching the goal.
3. **Completeness:** On `MAP_C`, UCS systematically explores the entire 13-node reachable component and terminates safely without getting stuck.

## 7. A* Search — Informed Search with Manhattan Distance Heuristic

### 7.1 Algorithm Overview and Heuristic Formulation
A* Search guides exploration toward the goal using the evaluation function:
$$f(n) = g(n) + h(n)$$
where $g(n)$ is the exact cost from the start to node $n$, and $h(n)$ is the estimated cost from node $n$ to the goal.

#### Manhattan Distance Heuristic
For 4-connected grid navigation, we calculate $h(n)$ as:
$$h(n) = |r_n - r_G| + |c_n - c_G|$$

#### Heuristic Properties
1. **Admissibility ($h(n) \le h^*(n)$):** Manhattan distance represents the ideal obstacle-free step count to $G$. Since every valid step costs at least $1$ ($c_{\min} = 1$), $h(n)$ never overestimates the true remaining cost, guaranteeing cost-optimal solutions.
2. **Consistency ($h(n) \le c(n, n') + h(n')$):** Moving to an adjacent cell changes Manhattan distance by at most 1 ($|h(n) - h(n')| \le 1 \le c(n, n')$). This monotonicity ensures $f$-values never decrease along a path and expanded nodes never need to be reopened.
"""

def manhattan_distance(p1: Coordinate, p2: Coordinate) -> int:
    """Calculates L1 Manhattan distance: |r1 - r2| + |c1 - c2|."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def a_star_search(grid: List[str], map_name: str = 'Grid') -> SearchResult:
    """
    Executes A* Search from Start to Goal utilizing Manhattan distance heuristic.
    Prioritizes frontier nodes by f(n) = g(n) + h(n).
    """
    grid_info = parse_and_validate_grid(grid)
    start_time = time.perf_counter()

    # Priority queue stores: (f_cost, sequence_id, current_coordinate, g_cost)
    frontier: List[Tuple[int, int, Coordinate, int]] = []
    counter = 0
    h_start = manhattan_distance(grid_info.start, grid_info.goal)
    heapq.heappush(frontier, (h_start, counter, grid_info.start, 0))

    # Track parent pointers and lowest known g(n) cost to each node
    came_from: Dict[Coordinate, Optional[Coordinate]] = {grid_info.start: None}
    cost_so_far: Dict[Coordinate, int] = {grid_info.start: 0}

    nodes_expanded = 0
    max_frontier_size = len(frontier)

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_f, _, current_node, current_g = heapq.heappop(frontier)

        # Skip stale entries if a cheaper g-cost path to this node was already found
        if current_g > cost_so_far[current_node]:
            continue

        nodes_expanded += 1

        # Goal check upon removal guarantees cost-optimality with admissible h
        if current_node == grid_info.goal:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path = reconstruct_path(came_from, grid_info.goal)
            return SearchResult(
                algorithm='A* Search',
                map_name=map_name,
                found=True,
                path=path,
                length=len(path) - 1,
                cost=cost_so_far[grid_info.goal],
                nodes_expanded=nodes_expanded,
                max_frontier_size=max_frontier_size,
                execution_time_ms=elapsed_ms,
            )

        # Evaluate neighbors using f(n') = g(n') + h(n')
        for neighbor, move_cost in get_valid_neighbors(current_node, grid_info):
            new_g = current_g + move_cost
            if neighbor not in cost_so_far or new_g < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_g
                came_from[neighbor] = current_node
                counter += 1
                h_cost = manhattan_distance(neighbor, grid_info.goal)
                f_cost = new_g + h_cost
                heapq.heappush(frontier, (f_cost, counter, neighbor, new_g))
                max_frontier_size = max(max_frontier_size, len(frontier))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm='A* Search',
        map_name=map_name,
        found=False,
        path=[],
        length=0,
        cost=float('inf'),
        nodes_expanded=nodes_expanded,
        max_frontier_size=max_frontier_size,
        execution_time_ms=elapsed_ms,
    )

"""### 7.2 A* Benchmark Results on MAP_A
---
"""

# Execute A* Search on MAP_A and validate solution
res_astar_a = a_star_search(MAP_A, map_name='MAP_A')
validate_path_solution(MAP_A, res_astar_a)

print('=== MAP_A: A* Search Results ===')
print(f'Status:             {res_astar_a.status}')
print(f'Path Coordinates:   {res_astar_a.path}')
print(f'Path Length (moves):{res_astar_a.length}')
print(f'Path Cost:          {res_astar_a.cost}')
print(f'Nodes Expanded:     {res_astar_a.nodes_expanded}')
print(f'Max Frontier Size:  {res_astar_a.max_frontier_size}')
print(f'Runtime (ms):       {res_astar_a.execution_time_ms:.3f} ms')
print('\nASCII Path Representation:')
print(render_ascii_path(MAP_A, res_astar_a.path))

fig_astar_a = plot_warehouse_grid(MAP_A, res_astar_a)
plt.show()

"""### 7.3 A* Benchmark Results on MAP_B — Guided Cost-Optimal Detour
---
"""

# Execute A* Search on MAP_B and validate solution
res_astar_b = a_star_search(MAP_B, map_name='MAP_B')
validate_path_solution(MAP_B, res_astar_b)

print('=== MAP_B: A* Search Results ===')
print(f'Status:             {res_astar_b.status}')
print(f'Path Coordinates:   {res_astar_b.path}')
print(f'Path Length (moves):{res_astar_b.length}')
print(f'Path Cost:          {res_astar_b.cost}')
print(f'Nodes Expanded:     {res_astar_b.nodes_expanded}')
print(f'Max Frontier Size:  {res_astar_b.max_frontier_size}')
print(f'Runtime (ms):       {res_astar_b.execution_time_ms:.3f} ms')
print('\nASCII Path Representation:')
print(render_ascii_path(MAP_B, res_astar_b.path))

fig_astar_b = plot_warehouse_grid(MAP_B, res_astar_b)
plt.show()

"""### 7.4 A* Benchmark Results on MAP_C — Unsolvable Termination Guarantee
---
"""

# Execute A* Search on MAP_C and validate unsolvable termination
res_astar_c = a_star_search(MAP_C, map_name='MAP_C')
validate_path_solution(MAP_C, res_astar_c)

astar_cost_c_str = f'{res_astar_c.cost:.0f}' if res_astar_c.found else 'N/A'
astar_len_c_str = f'{res_astar_c.length} moves' if res_astar_c.found else 'N/A'

print('=== MAP_C: A* Search Results ===')
print(f'Status:             {res_astar_c.status}')
if not res_astar_c.found:
    print('No path found')
print(f'Path Coordinates:   {res_astar_c.path}')
print(f'Path Length:        {astar_len_c_str}')
print(f'Path Cost:          {astar_cost_c_str}')
print(f'Nodes Expanded:     {res_astar_c.nodes_expanded}')
print(f'Max Frontier Size:  {res_astar_c.max_frontier_size}')
print(f'Runtime (ms):       {res_astar_c.execution_time_ms:.3f} ms')

fig_astar_c = plot_warehouse_grid(MAP_C, res_astar_c)
plt.show()

"""## 8. A* Search Summary Table & Interpretation

The table below aggregates the empirical results for A* Search across all benchmark maps:
"""

# Aggregate A* benchmark results into a summary DataFrame
astar_summary_data = [
    {
        'Map': res.map_name,
        'Algorithm': res.algorithm,
        'Status': res.status,
        'Path Length (Moves)': str(res.length) if res.found else 'N/A',
        'Path Cost': f'{res.cost:.0f}' if res.found else 'N/A',
        'Nodes Expanded': res.nodes_expanded,
        'Max Frontier Size': res.max_frontier_size,
        'Runtime (ms)': f'{res.execution_time_ms:.3f}'
    }
    for res in [res_astar_a, res_astar_b, res_astar_c]
]
df_astar_summary = pd.DataFrame(astar_summary_data)
df_astar_summary

"""### Interpretation of A* Performance
1. **Cost Optimality:** A* returns the exact same minimal path costs as UCS across all solvable maps.
2. **Search Efficiency:** Goal-directed heuristic pruning reduces state expansions substantially on solvable layouts.
3. **Completeness:** When no solution exists (`MAP_C`), A* safely visits the entire reachable component and terminates.

## 9. Final UCS vs. A* Comparative Analysis

The side-by-side comparison below contrasts both search algorithms across all evaluation metrics:
"""

# Build side-by-side comparative evaluation DataFrame
comparison_rows = [
    # MAP_A
    {'Map': 'MAP_A', 'Metric': 'Status', 'UCS': res_ucs_a.status, 'A*': res_astar_a.status, 'Difference / Reduction': 'Identical', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_A', 'Metric': 'Path Length (moves)', 'UCS': str(res_ucs_a.length), 'A*': str(res_astar_a.length), 'Difference / Reduction': '0 moves (0.0%)', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_A', 'Metric': 'Path Cost', 'UCS': f'{res_ucs_a.cost:.0f}', 'A*': f'{res_astar_a.cost:.0f}', 'Difference / Reduction': '0 cost (0.0%)', 'Better Algorithm': 'Tie (Optimal)'},
    {'Map': 'MAP_A', 'Metric': 'Nodes Expanded', 'UCS': str(res_ucs_a.nodes_expanded), 'A*': str(res_astar_a.nodes_expanded), 'Difference / Reduction': '-7 nodes (-25.0%)', 'Better Algorithm': 'A*'},
    {'Map': 'MAP_A', 'Metric': 'Max Frontier Size', 'UCS': str(res_ucs_a.max_frontier_size), 'A*': str(res_astar_a.max_frontier_size), 'Difference / Reduction': '+1 entry (+25.0%)', 'Better Algorithm': 'UCS'},
    {'Map': 'MAP_A', 'Metric': 'Runtime (ms)', 'UCS': f'{res_ucs_a.execution_time_ms:.3f}', 'A*': f'{res_astar_a.execution_time_ms:.3f}', 'Difference / Reduction': f'{res_astar_a.execution_time_ms - res_ucs_a.execution_time_ms:+.3f} ms', 'Better Algorithm': 'A*'},
    # MAP_B
    {'Map': 'MAP_B', 'Metric': 'Status', 'UCS': res_ucs_b.status, 'A*': res_astar_b.status, 'Difference / Reduction': 'Identical', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_B', 'Metric': 'Path Length (moves)', 'UCS': str(res_ucs_b.length), 'A*': str(res_astar_b.length), 'Difference / Reduction': '0 moves (0.0%)', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_B', 'Metric': 'Path Cost', 'UCS': f'{res_ucs_b.cost:.0f}', 'A*': f'{res_astar_b.cost:.0f}', 'Difference / Reduction': '0 cost (0.0%)', 'Better Algorithm': 'Tie (Optimal)'},
    {'Map': 'MAP_B', 'Metric': 'Nodes Expanded', 'UCS': str(res_ucs_b.nodes_expanded), 'A*': str(res_astar_b.nodes_expanded), 'Difference / Reduction': '-16 nodes (-51.6%)', 'Better Algorithm': 'A*'},
    {'Map': 'MAP_B', 'Metric': 'Max Frontier Size', 'UCS': str(res_ucs_b.max_frontier_size), 'A*': str(res_astar_b.max_frontier_size), 'Difference / Reduction': '+3 entries (+75.0%)', 'Better Algorithm': 'UCS'},
    {'Map': 'MAP_B', 'Metric': 'Runtime (ms)', 'UCS': f'{res_ucs_b.execution_time_ms:.3f}', 'A*': f'{res_astar_b.execution_time_ms:.3f}', 'Difference / Reduction': f'{res_astar_b.execution_time_ms - res_ucs_b.execution_time_ms:+.3f} ms', 'Better Algorithm': 'A*'},
    # MAP_C
    {'Map': 'MAP_C', 'Metric': 'Status', 'UCS': res_ucs_c.status, 'A*': res_astar_c.status, 'Difference / Reduction': 'Identical', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_C', 'Metric': 'Path Length (moves)', 'UCS': 'N/A', 'A*': 'N/A', 'Difference / Reduction': 'N/A', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_C', 'Metric': 'Path Cost', 'UCS': 'N/A', 'A*': 'N/A', 'Difference / Reduction': 'N/A', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_C', 'Metric': 'Nodes Expanded', 'UCS': str(res_ucs_c.nodes_expanded), 'A*': str(res_astar_c.nodes_expanded), 'Difference / Reduction': '0 nodes (0.0%)', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_C', 'Metric': 'Max Frontier Size', 'UCS': str(res_ucs_c.max_frontier_size), 'A*': str(res_astar_c.max_frontier_size), 'Difference / Reduction': '0 entries (0.0%)', 'Better Algorithm': 'Tie'},
    {'Map': 'MAP_C', 'Metric': 'Runtime (ms)', 'UCS': f'{res_ucs_c.execution_time_ms:.3f}', 'A*': f'{res_astar_c.execution_time_ms:.3f}', 'Difference / Reduction': f'{res_astar_c.execution_time_ms - res_ucs_c.execution_time_ms:+.3f} ms', 'Better Algorithm': 'Tie'},
]
df_comparison = pd.DataFrame(comparison_rows)
df_comparison

"""### Answers to Core Questions

#### 1. Which algorithm found the cheaper path?
**Both algorithms found identical, optimal path costs on all solvable maps** (Cost 15 on `MAP_A`, Cost 12 on `MAP_B`). Because all step costs are positive ($c \ge 1$) and the Manhattan heuristic is admissible ($h(n) \le h^*(n)$), both algorithms are mathematically guaranteed to return cost-optimal solutions.

#### 2. Which algorithm expanded fewer nodes?
**A* expanded fewer nodes on all solvable maps** (21 vs. 28 on `MAP_A`; 15 vs. 31 on `MAP_B`). The heuristic focuses exploration toward the goal, avoiding unproductive expansions in opposite directions. On unsolvable `MAP_C`, both algorithms expand all 13 reachable nodes.

#### 3. How does Maximum Frontier Size compare?
A*'s peak frontier was slightly larger on solvable maps (5 vs. 4 on `MAP_A`, 7 vs. 4 on `MAP_B`) because it discovers and queues multiple promising forward neighbors earlier in the search. This modest memory tradeoff is outweighed by the 25–51% reduction in expanded nodes.

### 9.1 Configurable Single-Map Demonstration (`GRID`)

This interactive cell executes both algorithms on whichever map is assigned to `GRID` at the top of the notebook (`GRID = MAP_A` by default), allowing standalone testing without modifying the benchmark suite.
"""

# Execute both algorithms on the user-configured active GRID variable
grid_name = 'MAP_A' if GRID == MAP_A else 'MAP_B' if GRID == MAP_B else 'MAP_C' if GRID == MAP_C else 'Custom GRID'

print(f'=== Single-Map Demonstration: Executing on Active GRID ({grid_name}) ===')
demo_res_ucs = uniform_cost_search(GRID, map_name=f'{grid_name} (Demo UCS)')
demo_res_astar = a_star_search(GRID, map_name=f'{grid_name} (Demo A*)')

print(f'UCS -> Status: {demo_res_ucs.status} | Cost: {demo_res_ucs.cost if demo_res_ucs.found else "N/A"} | Moves: {demo_res_ucs.length if demo_res_ucs.found else "N/A"} | Expanded: {demo_res_ucs.nodes_expanded}')
print(f'A*  -> Status: {demo_res_astar.status} | Cost: {demo_res_astar.cost if demo_res_astar.found else "N/A"} | Moves: {demo_res_astar.length if demo_res_astar.found else "N/A"} | Expanded: {demo_res_astar.nodes_expanded}')

# Render ASCII path for active GRID
if demo_res_astar.found:
    print(f'\nA* Path on {grid_name}:')
    print(render_ascii_path(GRID, demo_res_astar.path))
else:
    print(f'\nNo path found on {grid_name}.')

"""## 10. Final Comparison Visualizations

The bar charts below contrast **Search Efficiency** (Nodes Expanded) and **Memory Footprint** (Peak Frontier Size) across all benchmark maps.

> **Note on Runtime:** On small discrete grids ($5 \times 8$ and $5 \times 9$), execution runtimes are fractions of a millisecond ($< 0.25\text{ ms}$). Microsecond variations reflect background operating system scheduling, making node expansion count the primary metric of search efficiency.
"""

maps = ['MAP_A', 'MAP_B', 'MAP_C']
x = np.arange(len(maps))
width = 0.35

ucs_exp = [res_ucs_a.nodes_expanded, res_ucs_b.nodes_expanded, res_ucs_c.nodes_expanded]
astar_exp = [res_astar_a.nodes_expanded, res_astar_b.nodes_expanded, res_astar_c.nodes_expanded]

ucs_front = [res_ucs_a.max_frontier_size, res_ucs_b.max_frontier_size, res_ucs_c.max_frontier_size]
astar_front = [res_astar_a.max_frontier_size, res_astar_b.max_frontier_size, res_astar_c.max_frontier_size]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

# 1. Nodes Expanded (Search Efficiency: Lower is Better)
r1 = ax1.bar(x - width/2, ucs_exp, width, label='UCS (Uninformed)', color='#457B9D')
r2 = ax1.bar(x + width/2, astar_exp, width, label='A* (Informed)', color='#E76F51')
ax1.set_title('Search Efficiency: Nodes Expanded (Lower is Better)', fontweight='bold', fontsize=10.5, color='#1D3557')
ax1.set_ylabel('Expanded State Count')
ax1.set_xticks(x)
ax1.set_xticklabels(maps, fontweight='bold')
ax1.set_ylim(0, max(ucs_exp + astar_exp) * 1.25)
ax1.bar_label(r1, padding=3, fontsize=9)
ax1.bar_label(r2, padding=3, fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', framealpha=0.9)

# 2. Peak Frontier Size (Memory Footprint)
r3 = ax2.bar(x - width/2, ucs_front, width, label='UCS (Uninformed)', color='#457B9D')
r4 = ax2.bar(x + width/2, astar_front, width, label='A* (Informed)', color='#E76F51')
ax2.set_title('Memory Footprint: Peak Frontier Size', fontweight='bold', fontsize=10.5, color='#1D3557')
ax2.set_ylabel('Peak Queue Entries (States)')
ax2.set_xticks(x)
ax2.set_xticklabels(maps, fontweight='bold')
ax2.set_ylim(0, max(ucs_front + astar_front) * 1.30)
ax2.bar_label(r3, padding=3, fontsize=9)
ax2.bar_label(r4, padding=3, fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', framealpha=0.9)

plt.suptitle('Empirical Benchmark: Uniform Cost Search vs. A* Search', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

"""## 11. Algorithm Selection and Deployment Justification

### Recommendation: A* Search

Based on the empirical evidence and theoretical search properties, **A* Search with the Manhattan distance heuristic is the recommended algorithm for the warehouse delivery robot.**

#### Key Justifications
1. **Reduced Search Space:** A* reduces node expansions by **25.0%** on `MAP_A` (21 vs. 28) and **51.6%** on `MAP_B` (15 vs. 31), which scales significantly on larger real-world warehouse floors.
2. **Guaranteed Optimality:** Because the Manhattan distance is admissible and consistent on a 4-connected grid with $c_{\min}=1$, A* guarantees the exact same minimal path cost as UCS.
3. **Robust Termination:** When a goal is blocked (`MAP_C`), A* safely visits the reachable component and terminates without infinite looping.
4. **Practical Deployment:** Expanding fewer nodes reduces processor workload and allows faster dynamic replanning in embedded robot controllers.

## 12. Conclusion

This investigation highlights three primary findings:
1. **Shortest Step Count $\neq$ Lowest Traversal Cost:** On weighted grids like `MAP_B`, unweighted search algorithms that minimize step count pick an expensive 8-move route (cost 22). Both UCS and A* correctly identify the cost-optimal 12-move detour (cost 12).
2. **Heuristic Efficiency:** Incorporating an admissible Manhattan heuristic enables A* to achieve identical optimal paths while expanding up to **51.6% fewer nodes** than UCS.
3. **Completeness & Determinism:** The implemented algorithms handle obstacle avoidance, non-uniform terrain friction, and unsolvable barrier isolation deterministically and without external search libraries.
"""