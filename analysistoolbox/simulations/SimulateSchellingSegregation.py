# Load packages
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import textwrap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Declare function
def SimulateSchellingSegregation(tolerances=(0.0, 0.25, 0.5, 0.75, 1.0),
                                 # Population / grid parameters
                                 grid_shape=(25, 25),
                                 empty_fraction=0.1,
                                 number_of_factions=2,
                                 faction_fractions=None,
                                 starting_grid=None,
                                 # Neighborhood parameters
                                 neighborhood='moore',
                                 wrap_edges=False,
                                 # Simulation parameters
                                 max_steps=300,
                                 random_seed=412,
                                 # Tolerance sweep parameters
                                 include_tolerance_sweep=False,
                                 sweep_tolerances=None,
                                 sweep_number_of_seeds=3,
                                 # Output parameters
                                 return_format='dataframe',
                                 # Plotting parameters
                                 plot_simulation_results=True,
                                 faction_labels=None,
                                 faction_colors=None,
                                 empty_color="#FFFFFF",
                                 show_cell_borders=True,
                                 figure_size=None,
                                 # Text formatting arguments
                                 title_for_plot="Schelling Segregation Model",
                                 subtitle_for_plot="How individual tolerance for unlike neighbors shapes neighborhood segregation",
                                 caption_for_plot=None,
                                 data_source_for_plot=None):
    """
    Simulate Thomas Schelling's model of residential segregation across several tolerance levels.

    Agents belonging to two or more factions are scattered across a grid with some
    empty cells. Each agent looks at its neighbors; if the share of neighbors from a
    *different* faction exceeds the agent's tolerance, the agent is unhappy and moves
    to a random empty cell. The process repeats until every agent is happy or a step
    limit is reached. Running the same starting population under several tolerance
    levels side by side reveals how mild individual preferences can aggregate into
    strongly segregated collective patterns.

    Schelling-style agent-based simulations are useful for:
      * Urban Planning: Exploring how housing preferences can produce segregated neighborhoods without any central policy.
      * Sociology: Demonstrating that macro-level patterns need not reflect macro-level intentions.
      * Intelligence Analysis: Testing whether observed clustering of groups could arise from weak local preferences rather than coordination.
      * Public Health: Modeling how social sorting can concentrate exposure or access to care in particular areas.
      * Organizational Behavior: Simulating how team or office self-selection can silo departments.
      * Economics: Illustrating tipping points and path dependence in markets with local interactions.
      * Education: Teaching emergence, feedback loops, and agent-based modeling.

    Each step, every unhappy agent is relocated to a randomly chosen empty cell (in
    random order, for as long as empty cells last). An agent with no occupied
    neighbors counts as happy. The same random starting grid is used for every
    tolerance level so that differences between columns are due only to tolerance.

    Parameters
    ----------
    tolerances : sequence of float, optional
        The tolerance levels to compare, each between 0 and 1. Tolerance is the
        largest share of *unlike* neighbors an agent will accept before moving:
        0.0 means a single unlike neighbor is too many, 1.0 means any mix is fine
        (nobody ever moves). Schelling's "similarity threshold" is 1 - tolerance.
        Each tolerance gets its own column in the plot. Defaults to (0.0, 0.25, 0.5, 0.75, 1.0).
    grid_shape : tuple of int, optional
        The shape of the grid: (length,) for a 1-D row or (rows, columns) for a 2-D grid.
        Ignored if starting_grid is provided. Defaults to (25, 25).
    empty_fraction : float, optional
        The share of cells left empty (0 to less than 1). Agents can only move if
        empty cells exist. Ignored if starting_grid is provided. Defaults to 0.1.
    number_of_factions : int, optional
        The number of distinct factions (groups) of agents. Must be at least 2 and no
        more than 50% of the smaller grid dimension (or of the length of a 1-D row).
        Defaults to 2.
    faction_fractions : sequence of float, optional
        The share of occupied cells assigned to each faction. Must have one entry per
        faction and sum to 1. If None, occupied cells are split evenly. Ignored if
        starting_grid is provided. Defaults to None.
    starting_grid : array-like, optional
        A hand-built 1-D or 2-D array of integers to start from, where 0 is an empty
        cell and 1 to number_of_factions identify each faction. Overrides grid_shape,
        empty_fraction, and faction_fractions. Defaults to None.
    neighborhood : str, optional
        Which surrounding cells count as neighbors on a 2-D grid: 'moore' (the 8
        surrounding cells, including diagonals) or 'von_neumann' (the 4 cells sharing
        an edge). On a 1-D row both options use the 2 adjacent cells. Defaults to 'moore'.
    wrap_edges : bool, optional
        Whether the grid wraps around (a torus), so cells on one edge neighbor cells on
        the opposite edge. Requires every grid dimension to be at least 3. Defaults to False.
    max_steps : int, optional
        The maximum number of steps to run each simulation before stopping. Defaults to 300.
    random_seed : int, optional
        The seed for the random number generator to ensure replicability. Defaults to 412.
    include_tolerance_sweep : bool, optional
        Whether to also run the model across a fine range of tolerances (over several
        seeds) and add a panel showing final similarity versus tolerance. This runs
        many extra simulations. Defaults to False.
    sweep_tolerances : sequence of float, optional
        The tolerance levels used in the sweep. If None, uses 21 evenly spaced values
        from 0 to 1. Defaults to None.
    sweep_number_of_seeds : int, optional
        The number of random seeds averaged at each sweep tolerance. Defaults to 3.
    return_format : str, optional
        The format of the returned data: 'dataframe' (summary table only) or 'dict'
        (summary table plus grids, per-step histories, and sweep results).
        Defaults to 'dataframe'.
    plot_simulation_results : bool, optional
        Whether to display the combined figure of grids and histories. Defaults to True.
    faction_labels : sequence of str, optional
        Display names for each faction, used in the legend. If None, uses
        "Faction 1", "Faction 2", etc. Defaults to None.
    faction_colors : sequence of str, optional
        Hex color codes for each faction. If None, a colorblind-friendly palette is
        used for up to 10 factions and an evenly spaced hue palette beyond that.
        Defaults to None.
    empty_color : str, optional
        The hex color code for empty cells. Defaults to "#FFFFFF".
    show_cell_borders : bool, optional
        Whether to draw faint lines between grid cells. Defaults to True.
    figure_size : tuple, optional
        The size of the figure in inches (width, height). If None, it is sized
        automatically from the number of tolerances. Defaults to None.
    title_for_plot : str, optional
        The main title for the figure. Defaults to "Schelling Segregation Model".
    subtitle_for_plot : str, optional
        The descriptive subtitle for the figure.
        Defaults to "How individual tolerance for unlike neighbors shapes neighborhood segregation".
    caption_for_plot : str, optional
        Optional caption text displayed at the bottom of the figure. Defaults to None.
    data_source_for_plot : str, optional
        Optional data source identification text. Defaults to None.

    Returns
    -------
    pd.DataFrame or dict
        If return_format is 'dataframe', a summary table with one row per tolerance and
        the columns 'Tolerance', 'Similarity Threshold', 'Converged', 'Steps',
        'Initial Similarity', 'Final Similarity', and 'Final Unhappy Agents'.
        Similarity is the average share of an agent's neighbors that belong to its own
        faction. If return_format is 'dict', a dictionary with the keys:
          * 'summary': the summary table described above.
          * 'results': a dictionary keyed by tolerance, each holding the 'initial' and
            'final' grids (np.ndarray) and the per-step 'similarity_history' and
            'unhappy_history' lists.
          * 'sweep': a DataFrame of sweep results ('Tolerance', 'Seed',
            'Final Similarity'), or None if include_tolerance_sweep is False.

    Teaching Note
    -------------
    Schelling's central insight is that you cannot read individual motives off
    aggregate patterns. In his model, no agent wants a segregated neighborhood --
    an agent with a tolerance of 0.5 is perfectly content being in the minority
    half the time -- yet the population as a whole sorts itself into sharply
    divided clusters. Small, reasonable-seeming preferences, applied locally and
    repeatedly, compound through feedback: when one agent leaves, the faction
    balance of the neighborhood it left shifts, making the remaining agents more
    likely to leave as well. This is emergence: a collective outcome that no
    individual chose and that is far more extreme than any individual's preference.

    The model also shows that the relationship between preference and outcome is
    non-linear. At very high tolerance nobody moves and the grid stays mixed. At
    very low tolerance nobody can ever be satisfied, so agents churn endlessly
    without settling into stable clusters. The strongest segregation often appears
    at moderate-to-low tolerances, where enough agents settle to anchor clusters
    while the rest keep moving toward them. Comparing several tolerances side by
    side -- or sweeping across the full range -- reveals tipping points that any
    single scenario would hide.

    For analysts, the lesson is a caution against the "ecological fallacy" in
    reverse: observing a highly clustered pattern (of people, organizations, or
    activity) is not evidence of strong preferences or deliberate coordination.
    Weak local rules can be sufficient. Agent-based models like this one are a way
    to test whether a proposed micro-level mechanism is capable of producing the
    macro-level pattern you observe.

    Examples
    --------
    # Urban planning: compare five tolerance levels on a 25 x 25 neighborhood
    summary = SimulateSchellingSegregation(
        tolerances=(0.0, 0.25, 0.5, 0.75, 1.0),
        grid_shape=(25, 25),
        empty_fraction=0.1
    )

    # Sociology: three groups of unequal size on a wrapped grid, with a tolerance sweep
    results = SimulateSchellingSegregation(
        tolerances=(0.3, 0.5, 0.7),
        grid_shape=(40, 40),
        number_of_factions=3,
        faction_fractions=(0.5, 0.3, 0.2),
        faction_labels=('Majority', 'Minority 1', 'Minority 2'),
        wrap_edges=True,
        include_tolerance_sweep=True,
        return_format='dict'
    )

    # Teaching: a hand-built 1-D row that is small enough to check by hand
    row = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0]
    SimulateSchellingSegregation(
        tolerances=(0.0, 0.5, 1.0),
        starting_grid=row,
        max_steps=12
    )
    """

    # Ensure the tolerances are valid
    tolerances = [float(tol) for tol in np.atleast_1d(tolerances)]
    if len(tolerances) == 0:
        raise ValueError("Please provide at least one value in the tolerances argument.")
    if any(tol < 0 or tol > 1 for tol in tolerances):
        raise ValueError("Every value in the tolerances argument must be between 0 and 1.")

    # Ensure the number of factions is a whole number of at least 2
    if int(number_of_factions) != number_of_factions or number_of_factions < 2:
        raise ValueError("number_of_factions must be a whole number of at least 2.")
    number_of_factions = int(number_of_factions)

    # Ensure the grid shape (or starting grid) is valid
    if starting_grid is not None:
        starting_grid = np.asarray(starting_grid)
        if starting_grid.ndim not in (1, 2) or starting_grid.size == 0:
            raise ValueError("starting_grid must be a non-empty 1-D or 2-D array.")
        if not np.all(np.mod(starting_grid, 1) == 0):
            raise ValueError("starting_grid must contain only whole numbers (0 for empty, 1 to number_of_factions for each faction).")
        starting_grid = starting_grid.astype(int)
        if starting_grid.min() < 0 or starting_grid.max() > number_of_factions:
            raise ValueError("starting_grid values must be between 0 (empty) and number_of_factions (" + str(number_of_factions) + "). If your grid has more factions, increase number_of_factions.")
        grid_shape = starting_grid.shape
    else:
        grid_shape = tuple(np.atleast_1d(grid_shape))
        if len(grid_shape) not in (1, 2) or any(int(n) != n or n < 1 for n in grid_shape):
            raise ValueError("grid_shape must be (length,) for a 1-D row or (rows, columns) for a 2-D grid, using positive whole numbers.")
        grid_shape = tuple(int(n) for n in grid_shape)
        if empty_fraction < 0 or empty_fraction >= 1:
            raise ValueError("empty_fraction must be at least 0 and less than 1.")

    # Ensure the number of factions is no more than 50% of the grid's smallest dimension
    max_factions = int(0.5 * min(grid_shape))
    if number_of_factions > max_factions:
        raise ValueError("number_of_factions (" + str(number_of_factions) + ") must be no more than 50% of the grid's length or width, whichever is smaller. With a grid shape of " + str(grid_shape) + ", the maximum is " + str(max_factions) + ".")

    # Ensure the faction fractions are valid
    if faction_fractions is not None and starting_grid is None:
        faction_fractions = np.asarray(faction_fractions, dtype=float)
        if len(faction_fractions) != number_of_factions:
            raise ValueError("faction_fractions must have one value per faction (" + str(number_of_factions) + " values).")
        if np.any(faction_fractions < 0) or not np.isclose(faction_fractions.sum(), 1):
            raise ValueError("faction_fractions must be non-negative and sum to 1.")

    # Ensure the neighborhood and edge-wrapping arguments are valid
    if neighborhood not in ['moore', 'von_neumann']:
        raise ValueError("neighborhood must be either 'moore' or 'von_neumann'.")
    if wrap_edges and min(grid_shape) < 3:
        raise ValueError("wrap_edges requires every grid dimension to be at least 3.")

    # Ensure the simulation and output arguments are valid
    if int(max_steps) != max_steps or max_steps < 1:
        raise ValueError("max_steps must be a positive whole number.")
    if return_format not in ['dataframe', 'dict']:
        raise ValueError("return_format must be either 'dataframe' or 'dict'.")
    if include_tolerance_sweep:
        if sweep_tolerances is None:
            sweep_tolerances = np.linspace(0, 1, 21)
        sweep_tolerances = [float(tol) for tol in np.atleast_1d(sweep_tolerances)]
        if len(sweep_tolerances) == 0 or any(tol < 0 or tol > 1 for tol in sweep_tolerances):
            raise ValueError("sweep_tolerances must contain at least one value, and every value must be between 0 and 1.")
        if int(sweep_number_of_seeds) != sweep_number_of_seeds or sweep_number_of_seeds < 1:
            raise ValueError("sweep_number_of_seeds must be a positive whole number.")

    # Ensure the faction labels and colors match the number of factions
    if faction_labels is None:
        faction_labels = ["Faction " + str(i + 1) for i in range(number_of_factions)]
    elif len(faction_labels) != number_of_factions:
        raise ValueError("faction_labels must have one label per faction (" + str(number_of_factions) + " labels).")
    if faction_colors is None:
        default_colors = ["#3F7FBF", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                          "#D55E00", "#F0E442", "#0072B2", "#999999", "#882255"]
        if number_of_factions <= len(default_colors):
            faction_colors = default_colors[:number_of_factions]
        else:
            faction_colors = sns.color_palette("husl", number_of_factions).as_hex()
    elif len(faction_colors) != number_of_factions:
        raise ValueError("faction_colors must have one color per faction (" + str(number_of_factions) + " colors).")

    # Build the neighbor offsets for the chosen neighborhood
    ndim = len(grid_shape)
    if neighborhood == 'moore':
        offsets = [o for o in itertools.product((-1, 0, 1), repeat=ndim) if any(o)]
    else:
        offsets = [o for o in itertools.product((-1, 0, 1), repeat=ndim) if sum(map(abs, o)) == 1]

    # Count like and occupied neighbors for every cell
    def count_neighbors(grid):
        if wrap_edges:
            padded = np.pad(grid, 1, mode='wrap')
        else:
            padded = np.pad(grid, 1, constant_values=0)
        same = np.zeros(grid.shape, dtype=int)
        occupied = np.zeros(grid.shape, dtype=int)
        for offset in offsets:
            window = tuple(slice(1 + o, 1 + o + n) for o, n in zip(offset, grid.shape))
            neighbor = padded[window]
            occupied += neighbor != 0
            same += (neighbor == grid) & (neighbor != 0)
        return same, occupied

    # Flag agents whose share of unlike neighbors exceeds the tolerance
    def find_unhappy(grid, tolerance):
        same, occupied = count_neighbors(grid)
        unlike = occupied - same
        return (grid != 0) & (occupied > 0) & (unlike > tolerance * occupied + 1e-12)

    # Calculate the average share of like neighbors among agents with any neighbors
    def calculate_similarity(grid):
        same, occupied = count_neighbors(grid)
        has_neighbors = (grid != 0) & (occupied > 0)
        if not has_neighbors.any():
            return float("nan")
        return float((same[has_neighbors] / occupied[has_neighbors]).mean())

    # Create a random starting grid
    def create_random_grid(rng):
        number_of_cells = int(np.prod(grid_shape))
        number_empty = round(number_of_cells * empty_fraction)
        number_occupied = number_of_cells - number_empty
        if faction_fractions is None:
            shares = np.full(number_of_factions, 1 / number_of_factions)
        else:
            shares = faction_fractions
        # Allocate agents by largest remainder so faction counts sum to the occupied total
        raw_counts = shares * number_occupied
        counts = np.floor(raw_counts).astype(int)
        remainder_order = np.argsort(-(raw_counts - counts), kind='stable')
        counts[remainder_order[:number_occupied - counts.sum()]] += 1
        cells = np.concatenate([np.zeros(number_empty, dtype=int)] +
                               [np.full(count, faction + 1, dtype=int) for faction, count in enumerate(counts)])
        rng.shuffle(cells)
        return cells.reshape(grid_shape)

    # Run the model from a starting grid until nobody is unhappy or max_steps is reached
    def run_model(initial_grid, tolerance, rng):
        grid = initial_grid.copy()
        unhappy_history = []
        similarity_history = [calculate_similarity(grid)]
        converged = False
        for _ in range(int(max_steps)):
            movers = np.flatnonzero(find_unhappy(grid, tolerance))
            unhappy_history.append(len(movers))
            if len(movers) == 0:
                converged = True
                break
            # Move unhappy agents, in random order, to random empty cells
            empties = np.flatnonzero(grid == 0)
            rng.shuffle(movers)
            rng.shuffle(empties)
            number_moving = min(len(movers), len(empties))
            flat = grid.ravel().copy()
            flat[empties[:number_moving]] = flat[movers[:number_moving]]
            flat[movers[:number_moving]] = 0
            grid = flat.reshape(grid.shape)
            similarity_history.append(calculate_similarity(grid))
        return {
            "initial": initial_grid.copy(),
            "final": grid,
            "converged": converged,
            "steps": len(similarity_history) - 1,
            "similarity_history": similarity_history,
            "unhappy_history": unhappy_history,
            "final_unhappy": int(find_unhappy(grid, tolerance).sum()),
        }

    # Create a shared starting grid so that every tolerance starts from the same population
    rng = np.random.default_rng(random_seed)
    if starting_grid is not None:
        initial_grid = starting_grid.copy()
    else:
        initial_grid = create_random_grid(rng)

    # Run the model at each tolerance, each with its own reproducible random stream
    dict_results = {}
    for i, tolerance in enumerate(tolerances):
        tolerance_rng = np.random.default_rng([random_seed, i])
        dict_results[tolerance] = run_model(initial_grid, tolerance, tolerance_rng)

    # Summarize the results
    df_summary = pd.DataFrame({
        "Tolerance": tolerances,
        "Similarity Threshold": [1 - tol for tol in tolerances],
        "Converged": [dict_results[tol]["converged"] for tol in tolerances],
        "Steps": [dict_results[tol]["steps"] for tol in tolerances],
        "Initial Similarity": [dict_results[tol]["similarity_history"][0] for tol in tolerances],
        "Final Similarity": [dict_results[tol]["similarity_history"][-1] for tol in tolerances],
        "Final Unhappy Agents": [dict_results[tol]["final_unhappy"] for tol in tolerances],
    })

    # Run the tolerance sweep if requested
    df_sweep = None
    if include_tolerance_sweep:
        list_sweep_rows = []
        for seed_index in range(int(sweep_number_of_seeds)):
            seed = random_seed + seed_index
            sweep_rng = np.random.default_rng(seed)
            if starting_grid is not None:
                sweep_grid = starting_grid.copy()
            else:
                sweep_grid = create_random_grid(sweep_rng)
            for tolerance in sweep_tolerances:
                sweep_result = run_model(sweep_grid, tolerance, sweep_rng)
                list_sweep_rows.append({
                    "Tolerance": tolerance,
                    "Seed": seed,
                    "Initial Similarity": sweep_result["similarity_history"][0],
                    "Final Similarity": sweep_result["similarity_history"][-1],
                })
        df_sweep = pd.DataFrame(list_sweep_rows)

    # Generate plot if user requests it
    if plot_simulation_results:
        number_of_columns = len(tolerances)
        number_of_bottom_panels = 3 if include_tolerance_sweep else 2

        # Build a word-wrapped caption if one is provided
        wrapped_caption = None
        if caption_for_plot is not None or data_source_for_plot is not None:
            # Create starting point for caption
            wrapped_caption = ""

            # Add the caption to the plot, if one is provided
            if caption_for_plot is not None:
                # Word wrap the caption without splitting words
                wrapped_caption = textwrap.fill(caption_for_plot, 140, break_long_words=False)

            # Add the data source to the caption, if one is provided
            if data_source_for_plot is not None:
                wrapped_caption = wrapped_caption + "\n\nSource: " + data_source_for_plot
            wrapped_caption = wrapped_caption.strip("\n")

        # Reserve fixed space (in inches) for the header and footer
        header_height = 1.4
        footer_height = 0.6
        if wrapped_caption is not None:
            footer_height += 0.15 * (wrapped_caption.count("\n") + 1) + 0.1

        # Size the figure from the number of tolerances and the grid's aspect ratio
        is_one_dimensional = ndim == 1
        if figure_size is None:
            figure_width = max(10, 2.6 * number_of_columns)
            # Width of each grid panel after side margins (0.92) and spacing between panels (0.2)
            cell_width = 0.92 * figure_width / (number_of_columns + 0.2 * (number_of_columns - 1))
            if is_one_dimensional:
                grid_row_height = 0.9
            else:
                grid_row_height = min(cell_width * grid_shape[0] / grid_shape[1], 4)
            figure_size = (figure_width, header_height + 2 * grid_row_height + 4.3 + footer_height)
        else:
            grid_row_height = 1 if is_one_dimensional else 2.4
        figure_height = figure_size[1]

        # Create figure and layout: initial grids, final grids, and history charts
        fig = plt.figure(figsize=figure_size)
        outer_layout = fig.add_gridspec(
            nrows=3,
            ncols=number_of_columns,
            height_ratios=[grid_row_height, grid_row_height, 3],
            hspace=0.65 / ((2 * grid_row_height + 3) / 3),
            top=1 - header_height / figure_height,
            bottom=footer_height / figure_height,
            wspace=0.2,
            left=0.06,
            right=0.98
        )
        color_map = ListedColormap([empty_color] + list(faction_colors))

        # Draw the initial and final grids for each tolerance
        for column, tolerance in enumerate(tolerances):
            result = dict_results[tolerance]
            for row, key in enumerate(("initial", "final")):
                ax = fig.add_subplot(outer_layout[row, column])
                grid_to_show = np.atleast_2d(result[key])
                ax.imshow(
                    grid_to_show,
                    cmap=color_map,
                    vmin=0,
                    vmax=number_of_factions,
                    interpolation='nearest',
                    aspect='auto' if is_one_dimensional else 'equal'
                )
                # Add faint lines between cells
                if show_cell_borders:
                    ax.set_xticks(np.arange(-0.5, grid_to_show.shape[1], 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, grid_to_show.shape[0], 1), minor=True)
                    ax.grid(which='minor', color="#D9D9D9", linewidth=0.3)
                ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
                for spine in ax.spines.values():
                    spine.set_color("#D9D9D9")
                    spine.set_linewidth(0.5)
                # Label the column with its tolerance, and the final grid with its outcome
                if key == "initial":
                    ax.set_title(
                        "Tolerance {:.0%}".format(tolerance),
                        fontname="Arial",
                        fontsize=11,
                        color="#262626",
                        loc='left'
                    )
                else:
                    if result["converged"]:
                        outcome = "Settled after {} steps".format(result["steps"])
                    else:
                        outcome = "Not settled after {} steps".format(result["steps"])
                    ax.set_title(
                        outcome + "\nSimilarity {:.2f} → {:.2f}".format(
                            result["similarity_history"][0],
                            result["similarity_history"][-1]
                        ),
                        fontname="Arial",
                        fontsize=8,
                        color="#666666",
                        loc='left'
                    )
                # Label the rows on the first column
                if column == 0:
                    ax.set_ylabel(
                        key.capitalize(),
                        fontname="Arial",
                        fontsize=10,
                        color="#666666"
                    )

        # Create the bottom row of history (and optional sweep) charts
        bottom_layout = outer_layout[2, :].subgridspec(1, number_of_bottom_panels, wspace=0.3)
        line_colors = plt.cm.viridis(np.linspace(0, 0.9, number_of_columns))
        ax_similarity = fig.add_subplot(bottom_layout[0, 0])
        ax_unhappy = fig.add_subplot(bottom_layout[0, 1])
        for line_color, tolerance in zip(line_colors, tolerances):
            ax_similarity.plot(
                dict_results[tolerance]["similarity_history"],
                color=line_color,
                linewidth=1.5,
                label="{:.0%}".format(tolerance)
            )
            ax_unhappy.plot(
                dict_results[tolerance]["unhappy_history"],
                color=line_color,
                linewidth=1.5,
                label="{:.0%}".format(tolerance)
            )
        ax_similarity.set_ylim(0, 1.05)
        max_unhappy = max(max(dict_results[tol]["unhappy_history"]) for tol in tolerances)
        ax_unhappy.set_ylim(0, max(max_unhappy, 1) * 1.35)
        list_bottom_axes = [
            (ax_similarity, "Similarity over time", "Step", "Avg. share of like neighbors"),
            (ax_unhappy, "Unhappy agents over time", "Step", "Count"),
        ]

        # Add the tolerance sweep panel if requested
        if include_tolerance_sweep:
            ax_sweep = fig.add_subplot(bottom_layout[0, 2])
            df_sweep_summary = df_sweep.groupby("Tolerance")["Final Similarity"].agg(['mean', 'min', 'max']).reset_index()
            ax_sweep.fill_between(
                df_sweep_summary["Tolerance"],
                df_sweep_summary["min"],
                df_sweep_summary["max"],
                color="#262626",
                alpha=0.12,
                linewidth=0
            )
            ax_sweep.plot(
                df_sweep_summary["Tolerance"],
                df_sweep_summary["mean"],
                color="#262626",
                marker="o",
                markersize=3,
                linewidth=1.5
            )
            # Show the similarity of the random (unmoved) starting grids for reference
            baseline_similarity = df_sweep["Initial Similarity"].mean()
            ax_sweep.axhline(
                y=baseline_similarity,
                color="#262626",
                linestyle="--",
                linewidth=1,
                alpha=0.5
            )
            ax_sweep.text(
                x=1,
                y=baseline_similarity - 0.03,
                s="Random mix",
                horizontalalignment='right',
                verticalalignment='top',
                fontname="Arial",
                fontsize=8,
                color="#666666"
            )
            ax_sweep.set_ylim(0, 1.05)
            ax_sweep.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))
            list_bottom_axes.append(
                (ax_sweep, "Final similarity by tolerance", "Tolerance", "Avg. share of like neighbors")
            )

        # Format the bottom row of charts
        for ax, chart_title, x_label, y_label in list_bottom_axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#262626")
            ax.spines['bottom'].set_color("#262626")
            ax.set_title(chart_title, fontname="Arial", fontsize=11, color="#262626", loc='left')
            ax.set_xlabel(x_label, fontname="Arial", fontsize=9, color="#666666")
            ax.set_ylabel(y_label, fontname="Arial", fontsize=9, color="#666666")
            ax.tick_params(axis='both', which='major', labelsize=8, labelcolor="#666666")
            for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
                tick_label.set_fontname("Arial")
        ax_unhappy.legend(
            title="Tolerance",
            title_fontsize=8,
            fontsize=8,
            frameon=False,
            loc='upper right',
            ncol=min(number_of_columns, 5),
            columnspacing=1,
            handlelength=1.5
        )

        # Set the title and subtitle at the top of the figure
        fig.text(
            x=0.06,
            y=1 - 0.35 / figure_height,
            s=title_for_plot,
            fontname="Arial",
            fontsize=14,
            color="#262626"
        )
        fig.text(
            x=0.06,
            y=1 - 0.62 / figure_height,
            s=subtitle_for_plot,
            fontname="Arial",
            fontsize=11,
            color="#666666"
        )

        # Add a legend for the factions below the subtitle
        legend_handles = [Patch(facecolor=color, edgecolor="#D9D9D9", label=label)
                          for label, color in zip(faction_labels, faction_colors)]
        legend_handles.append(Patch(facecolor=empty_color, edgecolor="#D9D9D9", label="Empty"))
        fig.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.055, 1 - 0.7 / figure_height),
            ncol=min(len(legend_handles), 8),
            frameon=False,
            fontsize=9,
            handlelength=1,
            handleheight=1,
            columnspacing=1.2
        )

        # Add the caption to the bottom of the figure
        if wrapped_caption is not None:
            fig.text(
                x=0.06,
                y=0.15 / figure_height,
                s=wrapped_caption,
                fontname="Arial",
                fontsize=8,
                color="#666666",
                verticalalignment='bottom'
            )

        # Show plot
        plt.show()

        # Clear plot
        plt.clf()

    # Return the results
    if return_format == 'dataframe':
        return df_summary
    else:
        return {
            "summary": df_summary,
            "results": dict_results,
            "sweep": df_sweep,
        }
