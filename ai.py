
"""
Project 3: Simple Delivery Allocation (Load Balancing)
Two vehicles share 6 delivery packages from one depot.
BFS finds the true shortest distance from the depot to each point.
A Genetic Algorithm (GA) then evolves the best binary assignment
(Vehicle 1 vs Vehicle 2) so their total workloads are as equal
as possible.
Libraries: numpy, random, matplotlib  (no AI libraries)
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# -------------------------------------------------------------------------
# 1. ENVIRONMENT
# -------------------------------------------------------------------------
GRID_ROWS = 10
GRID_COLS = 10
# 0 = free path, 1 = obstacle
grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
OBSTACLES = [
    (1, 3), (2, 3), (3, 3),
    (5, 5), (5, 6), (5, 7),
    (7, 2), (8, 2),
    (6, 8), (7, 8),
]
for row, col in OBSTACLES:
    grid[row][col] = 1

DEPOT = (0, 0)
DELIVERY_POINTS = {
    "P1": (1, 7),
    "P2": (3, 8),
    "P3": (4, 1),
    "P4": (6, 4),
    "P5": (8, 6),
    "P6": (9, 9),
}

# GA hyper-parameters
POPULATION_SIZE = 100
GENERATIONS     = 300
MUTATION_RATE   = 0.15
TOURNAMENT_SIZE = 5
NUM_PACKAGES    = len(DELIVERY_POINTS)  # 6

# -------------------------------------------------------------------------
# 2. BFS — shortest path cost (number of steps) ignoring edge weights
# -------------------------------------------------------------------------
def bfs(grid, start, goal):
    """
    Return the fewest steps from start to goal on grid.
    Cells with value 1 are walls and cannot be entered.
    Returns -1 when no path exists.
    """
    rows, cols = grid.shape
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue   = deque([(start, 0)])   # (position, distance_so_far)
    visited = {start}

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) == goal:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows and 0 <= ny < cols
                    and grid[nx][ny] == 0
                    and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

    return -1   # no path found

def calculate_depot_distances(grid, depot, delivery_points):
    """Run BFS from the depot to every delivery point and return a dict."""
    return {name: bfs(grid, depot, loc)
            for name, loc in delivery_points.items()}

# -------------------------------------------------------------------------
# 3. GA — binary chromosome for load balancing
# -------------------------------------------------------------------------
# ---- chromosome helpers ----------------------------------------------------
def random_chromosome(length):
    """Binary chromosome: 0 → Vehicle 1, 1 → Vehicle 2."""
    return [random.randint(0, 1) for _ in range(length)]

def fitness(chromosome, distances):
    """
    Fitness = 1 / (1 + |total_v1 - total_v2|)
    Perfect balance (difference = 0) → fitness = 1.0  (maximum)
    Large imbalance           → fitness → 0  (minimum)

    We add 1 in the denominator to avoid division-by-zero and to
    keep fitness in the half-open range (0, 1].
    """
    total_v1 = sum(d for gene, d in zip(chromosome, distances) if gene == 0)
    total_v2 = sum(d for gene, d in zip(chromosome, distances) if gene == 1)
    imbalance = abs(total_v1 - total_v2)
    return 1.0 / (1.0 + imbalance)

# ---- selection -------------------------------------------------------------
def tournament_selection(population, fitnesses, k):
    """Pick k random candidates and return the best one."""
    candidates = random.sample(range(len(population)), k)
    best = max(candidates, key=lambda i: fitnesses[i])
    return population[best]

# ---- crossover -------------------------------------------------------------
def single_point_crossover(parent_a, parent_b):
    """
    Single-point crossover.
    A random cut-point splits each parent; the halves are swapped
    to produce two offspring.
    """
    if len(parent_a) <= 1:
        return parent_a[:], parent_b[:]
    cut = random.randint(1, len(parent_a) - 1)
    child_a = parent_a[:cut] + parent_b[cut:]
    child_b = parent_b[:cut] + parent_a[cut:]
    return child_a, child_b

# ---- mutation --------------------------------------------------------------
def bit_flip_mutation(chromosome, rate):
    """
    Flip each gene independently with probability rate.
    Bit-flip is the standard mutation for binary chromosomes.
    """
    return [1 - gene if random.random() < rate else gene
            for gene in chromosome]

# ---- main GA loop ----------------------------------------------------------
def run_ga(distances, population_size, generations, mutation_rate, tournament_size):
    """
    Evolve a population of binary chromosomes.
    Returns
    -------
    best_chromosome : list   – best assignment found
    best_fitness    : float  – its fitness score
    history         : list   – best fitness per generation (for plotting)
    """
    num_packages = len(distances)

    # --- initialise population ----------------------------------------------
    population = [random_chromosome(num_packages)
                  for _ in range(population_size)]

    best_chromosome = None
    best_fitness_val = -1.0
    history = []

    for generation in range(generations):

        # evaluate every chromosome
        fitnesses = [fitness(chrom, distances) for chrom in population]

        # track the overall best
        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fitness_val:
            best_fitness_val  = fitnesses[gen_best_idx]
            best_chromosome   = population[gen_best_idx][:]

        history.append(best_fitness_val)

        # --- elitism: carry the current best into the next generation -------
        new_population = [best_chromosome[:]]

        # --- fill the rest of the new population ----------------------------
        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, tournament_size)
            parent_b = tournament_selection(population, fitnesses, tournament_size)
            child_a, child_b = single_point_crossover(parent_a, parent_b)
            new_population.append(bit_flip_mutation(child_a, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(bit_flip_mutation(child_b, mutation_rate))

        population = new_population

    return best_chromosome, best_fitness_val, history

# -------------------------------------------------------------------------
# 4. RESULTS REPORTING
# -------------------------------------------------------------------------
def report_allocation(chromosome, distances, delivery_points):
    """Print a formatted summary of the final vehicle allocation."""
    names      = list(delivery_points.keys())
    dist_vals  = [distances[n] for n in names]
    v1_names = [n for n, g in zip(names, chromosome) if g == 0]
    v2_names = [n for n, g in zip(names, chromosome) if g == 1]

    total_v1 = sum(distances[n] for n in v1_names)
    total_v2 = sum(distances[n] for n in v2_names)

    print("\n" + "=" * 50)
    print("       DELIVERY ALLOCATION RESULT ")
    print("=" * 50)

    print(f"\nVehicle 1  →  packages: {v1_names} ")
    for n in v1_names:
        print(f"   {n}  location {delivery_points[n]}  distance = {distances[n]} ")
    print(f"   TOTAL distance = {total_v1} ")

    print(f"\nVehicle 2  →  packages: {v2_names} ")
    for n in v2_names:
        print(f"   {n}  location {delivery_points[n]}  distance = {distances[n]} ")
    print(f"   TOTAL distance = {total_v2} ")

    print(f"\nImbalance (|V1 - V2|) = {abs(total_v1 - total_v2)} ")
    print("=" * 50)

# -------------------------------------------------------------------------
# 5. VISUALISATION
# -------------------------------------------------------------------------
def visualize(grid, depot, delivery_points, distances, chromosome, history):
    """
    Two-panel matplotlib figure:
    Left  – grid map with depot, delivery points coloured by vehicle
    Right – GA fitness convergence curve
    """
    names     = list(delivery_points.keys())
    locations = list(delivery_points.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Project 3 – Delivery Allocation (Load Balancing)",
                 fontsize=14, fontweight="bold")

    # ---- left panel: grid map ----------------------------------------------
    ax_map = axes[0]
    ax_map.set_title("Grid Map & Vehicle Assignment")

    # Draw grid background: grey = wall, white = free
    map_display = np.ones((GRID_ROWS, GRID_COLS, 3))   # white
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 1:
                map_display[r, c] = [0.3, 0.3, 0.3]   # dark grey wall

    ax_map.imshow(map_display, origin="upper")
    ax_map.set_xticks(range(GRID_COLS))
    ax_map.set_yticks(range(GRID_ROWS))
    ax_map.grid(color="lightgrey", linewidth=0.5)

    # Depot marker
    ax_map.plot(depot[1], depot[0],  "b^", markersize=14, label="Depot")
    ax_map.annotate("Depot", xy=(depot[1], depot[0]),
                    xytext=(depot[1] + 0.4, depot[0] - 0.4),
                    fontsize=8, color="blue")

    # Delivery point markers coloured by assigned vehicle
    v1_total = 0
    v2_total = 0
    for name, loc, gene in zip(names, locations, chromosome):
        col   = "green"  if gene == 0 else  "orange"
        label = "Vehicle 1 (green)" if gene == 0 else  "Vehicle 2 (orange)"

        ax_map.plot(loc[1], loc[0],  "o", color=col, markersize=12)
        ax_map.annotate(
            f"{name}\n(d={distances[name]})",
            xy=(loc[1], loc[0]),
            xytext=(loc[1] + 0.35, loc[0] - 0.45),
            fontsize=7.5,
            color=col,
            fontweight="bold",
        )

        # Line from depot to delivery point
        ax_map.plot(
            [depot[1], loc[1]], [depot[0], loc[0]],
            color=col, linewidth=1.2, linestyle="--", alpha=0.6,
        )

        if gene == 0:
            v1_total += distances[name]
        else:
            v2_total += distances[name]

    # Legend & distance summary
    patch_v1 = mpatches.Patch(color="green",  label=f"Vehicle 1 (total = {v1_total})")
    patch_v2 = mpatches.Patch(color="orange", label=f"Vehicle 2 (total = {v2_total})")
    depot_h  = mpatches.Patch(color="blue",   label="Depot")
    ax_map.legend(handles=[depot_h, patch_v1, patch_v2],
                  loc="lower right", fontsize=8)

    # ---- right panel: convergence curve ------------------------------------
    ax_ga = axes[1]
    ax_ga.set_title("GA Fitness Convergence")
    ax_ga.plot(history, color="royalblue", linewidth=1.5)
    ax_ga.set_xlabel("Generation")
    ax_ga.set_ylabel("Best Fitness [1 / (1 + imbalance)]")
    ax_ga.set_ylim(0, 1.05)
    ax_ga.axhline(y=1.0, color="green", linestyle="--",
                  linewidth=1, label="Perfect balance (fitness = 1.0)")
    ax_ga.legend(fontsize=8)
    ax_ga.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("delivery_allocation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved → delivery_allocation.png")

# -------------------------------------------------------------------------
# 6. MAIN
# -------------------------------------------------------------------------
def main():
    # --- BFS phase ----------------------------------------------------------
    print("Running BFS to compute depot → delivery point distances …")
    distances = calculate_depot_distances(grid, DEPOT, DELIVERY_POINTS)

    print("\nShortest distances from depot:")
    for name, dist in distances.items():
        status = str(dist) if dist != -1 else "unreachable"
        print(f"  {name}  {DELIVERY_POINTS[name]}  →  {status} steps")

    # Ordered list matching DELIVERY_POINTS key order
    dist_list = [distances[n] for n in DELIVERY_POINTS]

    # --- GA phase -----------------------------------------------------------
    print(f"\nRunning GA (population={POPULATION_SIZE}, "
          f"generations={GENERATIONS}, "
          f"mutation_rate={MUTATION_RATE}) …")

    random.seed(42)   # reproducibility
    best_chromosome, best_fit, history = run_ga(
        distances       = dist_list,
        population_size = POPULATION_SIZE,
        generations     = GENERATIONS,
        mutation_rate   = MUTATION_RATE,
        tournament_size = TOURNAMENT_SIZE,
    )

    print(f"Best fitness achieved: {best_fit:.4f}")
    print(f"Best chromosome:       {best_chromosome}")

    # --- report & visualise -------------------------------------------------
    report_allocation(best_chromosome, distances, DELIVERY_POINTS)
    visualize(grid, DEPOT, DELIVERY_POINTS, distances,
               best_chromosome, history)

if __name__ == "__main__":
    main()