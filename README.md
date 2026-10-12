# Autonomous Warehouse Delivery Robot: Pathfinding with UCS vs. A*

A from-scratch Python implementation and benchmark of two discrete graph-search algorithms, **Uniform Cost Search (UCS)** and **A\* Search**, applied to an autonomous warehouse delivery robot navigating a 2D grid with obstacles and variable floor friction.

> Course project for **Intelligent Systems / Artificial Intelligence**. No external search libraries are used.

---

## Problem Statement

A robot must deliver inventory from a start dock (`S`) to a goal (`G`) across a warehouse floor. Rough floor patches cost more to cross than normal tiles, so the path with the fewest steps is **not necessarily** the cheapest route. The goal is to find the minimum-cost path and compare how efficiently each algorithm finds it.

### Grid Legend and Cost Model

| Symbol | Meaning | Entering Cost |
| :---: | :--- | :---: |
| `S` | Start dock | 0 |
| `G` | Goal dock | 1 |
| `.` | Normal floor | 1 |
| `~` | Rough floor (high friction / debris) | 3 |
| `#` | Obstacle / wall | impassable |

- **State space:** grid coordinates `(row, col)`
- **Actions:** 4 orthogonal moves (up, down, left, right); no diagonals
- **Step cost:** determined by the tile being **entered**

---

## Algorithms

### Uniform Cost Search (UCS), uninformed
Expands nodes in order of accumulated path cost `g(n)` using a priority queue. Complete and cost-optimal for positive step costs.

### A\* Search, informed
Expands nodes in order of `f(n) = g(n) + h(n)`, using the **Manhattan distance** heuristic:

```
h(n) = |r_n - r_G| + |c_n - c_G|
```

The heuristic is **admissible** (never overestimates, since every step costs at least 1) and **consistent**, so A\* returns cost-optimal paths while exploring fewer nodes.

### Implementation highlights
- Priority queue via `heapq` with an insertion counter for deterministic tie-breaking
- Goal test on **pop** (not on push) to guarantee optimality
- Stale-entry skipping (lazy deletion) instead of decrease-key
- Parent-pointer path reconstruction
- Grid validation (rectangular shape, valid characters, exactly one `S` and one `G`)
- Automated solution verification (path continuity, obstacle avoidance, cost recomputation)

---

## Benchmark Maps

| Map | Purpose |
| :--- | :--- |
| `MAP_A` | Standard warehouse floor with wall pillars and rough patches |
| `MAP_B` | Cost-vs-steps tradeoff: a short rough route (8 moves, cost 22) versus a longer clean detour (12 moves, cost 12) |
| `MAP_C` | Unbroken wall barrier; the goal is unreachable (tests termination) |

---

## Results

| Map | Algorithm | Status | Moves | Cost | Nodes Expanded | Peak Frontier |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| MAP_A | UCS | Success | 11 | 15 | 28 | 4 |
| MAP_A | A\* | Success | 11 | 15 | **21** | 5 |
| MAP_B | UCS | Success | 12 | 12 | 31 | 4 |
| MAP_B | A\* | Success | 12 | 12 | **15** | 7 |
| MAP_C | UCS | No path found | N/A | N/A | 13 | 3 |
| MAP_C | A\* | No path found | N/A | N/A | 13 | 3 |

### Key Findings
1. **Shortest in steps does not mean cheapest.** On `MAP_B`, a step-minimizing search would take the 8-move route costing 22. Both UCS and A\* correctly choose the 12-move detour costing 12.
2. **Both algorithms are cost-optimal.** They return identical path costs on every solvable map.
3. **A\* explores less.** It expands 25% fewer nodes on `MAP_A` and 51.6% fewer on `MAP_B`.
4. **A\* uses slightly more frontier memory.** It queues promising forward neighbors earlier, a small tradeoff for the reduction in expansions.
5. **Unsolvable maps terminate cleanly.** Both algorithms exhaust the 13 reachable cells on `MAP_C` and report that no path exists.

**Recommendation:** A\* with the Manhattan heuristic for deployment, since it gives the same optimal paths with less search effort.

> Runtimes on these small grids are fractions of a millisecond and dominated by timing noise, so nodes expanded is used as the primary efficiency metric.

---

## Evaluation Metrics

Each run records: **path length** (moves), **path cost**, **nodes expanded**, **maximum frontier size**, and **execution time** (`time.perf_counter()`).

---

## Getting Started

### Requirements
- Python 3.8+
- `numpy`, `pandas`, `matplotlib`
- Jupyter Notebook or JupyterLab

```bash
pip install numpy pandas matplotlib jupyter
```

### Run

```bash
git clone https://github.com/shaheenmohammedsh-ops/warehouse-delivery-robot-search-algorithms.git
cd warehouse-delivery-robot-search-algorithms
jupyter notebook Warehouse_Delivery_Robot_Search_Algorithms.ipynb
```

Then select **Run All**.

### Try your own map
Edit the `GRID` variable in the setup cell to point at your own list of strings (it must be rectangular and contain exactly one `S` and one `G`), then run Section 9.1 to compare both algorithms on it:

```python
GRID = [
    "S..#....",
    "##.#.##.",
    "....~~..",
    ".####.#.",
    "...~..#G",
]
```

---

## Repository Structure

```
.
├── Warehouse_Delivery_Robot_Search_Algorithms.ipynb   # Full implementation, benchmarks, and analysis
└── README.md
```

## Notebook Contents

1. Project overview and problem statement
2. Problem formulation and cost model
3. Environment setup and benchmark maps
4. Shared data structures, validation, and visualization
5. Uniform Cost Search (implementation and results)
6. A\* Search (implementation and results)
7. Side-by-side comparison and analysis
8. Configurable single-map demonstration
9. Comparison charts (nodes expanded, peak frontier)
10. Algorithm selection and conclusion

## Technologies

Python · heapq · NumPy · pandas · Matplotlib · Jupyter

## Possible Extensions

- Tighter heuristics that account for rough-tile costs
- Larger grids and randomly generated warehouse layouts for scaling tests
- Diagonal movement and dynamic obstacles
- Additional algorithms (Dijkstra with early exit, Greedy Best-First, Weighted A\*, IDA\*)

---



