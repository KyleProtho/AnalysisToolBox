# Load packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Declare function
def SimulateStandingOvationModel(qualities=(0.4, 0.5, 0.6, 0.7),
                                 # Audience perception parameters
                                 quality_threshold=0.6,
                                 noise=0.1,
                                 # Peer pressure parameters
                                 peer_threshold=0.5,
                                 peer_spread=0.0,
                                 can_sit_back_down=False,
                                 # Hall / sight line parameters
                                 hall_shape=(20, 40),
                                 neighborhood='cone',
                                 cone_depth=3,
                                 # Pioneer parameters
                                 number_of_pioneers=0,
                                 pioneer_placement='front',
                                 # Simulation parameters
                                 max_steps=50,
                                 ovation_level=0.8,
                                 random_seed=412,
                                 print_step_by_step=False,
                                 # Quality sweep parameters
                                 include_quality_sweep=False,
                                 sweep_qualities=None,
                                 sweep_number_of_seeds=10,
                                 # Output parameters
                                 return_format='dataframe',
                                 # Plotting parameters
                                 plot_simulation_results=True,
                                 state_colors=None,
                                 show_cell_borders=True,
                                 figure_size=None,
                                 # Text formatting arguments
                                 title_for_plot="Standing Ovation Model",
                                 subtitle_for_plot="How private judgment and peer pressure combine to decide whether an audience stands",
                                 caption_for_plot=None,
                                 data_source_for_plot=None):
    """
    Simulate Miller and Page's standing ovation model across several show qualities.

    An audience sits in a hall of rows and seats, with row 0 at the front, closest
    to the stage. When the curtain falls, each person decides whether to stand in
    two phases:

      * Step 0 (own judgment): each agent receives a private signal of the show's
        quality, equal to the true quality plus a Normal(0, noise) error. They stand
        if their signal is greater than quality_threshold. Pioneers stand no matter what.
      * Step 1+ (peer pressure): each agent looks at the seats they can see. A seated
        agent stands if more than their personal peer threshold of those people are
        standing. With can_sit_back_down, a standing (non-pioneer) agent sits again if
        more than their peer threshold of the people they see are seated. Everyone
        updates at the same time, and the run stops when nobody changes.

    Running the same audience (same perception errors, same peer thresholds, same
    pioneers) under several show qualities side by side reveals how peer pressure
    amplifies, or fails to amplify, private opinion into a collective outcome.

    Standing ovation style agent-based simulations are useful for:
      * Marketing: Exploring how early adopters and visible endorsements tip product adoption.
      * Sociology: Demonstrating how conformity turns a lukewarm crowd into a unanimous one.
      * Intelligence Analysis: Testing whether apparent consensus reflects shared judgment or information cascades.
      * Political Science: Modeling how visible public support influences undecided voters.
      * Organizational Behavior: Simulating how meeting seating and visibility shape group decisions.
      * Finance: Illustrating herding, where investors follow the actions they can observe.
      * Education: Teaching emergence, thresholds, and the role of network structure.

    Parameters
    ----------
    qualities : sequence of float, optional
        The true show quality values to compare. Each quality gets its own column in
        the plot, and every column uses the same audience. Defaults to (0.4, 0.5, 0.6, 0.7).
    quality_threshold : float, optional
        How good the show has to seem before an agent stands on their own judgment.
        Defaults to 0.6.
    noise : float, optional
        The standard deviation of each agent's perception error. Higher values mean
        more diversity of taste in the audience. Must be at least 0. Defaults to 0.1.
    peer_threshold : float, optional
        The share of visible people who must be standing before a seated agent stands,
        between 0 and 1. 0 means an agent stands if anyone they see stands; 1 means
        agents never give in to peer pressure. Defaults to 0.5.
    peer_spread : float, optional
        Each agent's peer threshold is drawn uniformly from peer_threshold ± peer_spread
        (clipped to 0 to 1), adding diversity of conformity. Must be at least 0.
        Defaults to 0.0.
    can_sit_back_down : bool, optional
        Whether a standing agent (other than a pioneer) sits down again when more than
        their peer threshold of the people they see are seated. Defaults to False.
    hall_shape : tuple of int, optional
        The shape of the hall as (rows, seats per row). Row 0 is the front. Defaults to (20, 30).
    neighborhood : str, optional
        Which seats an agent can see:
          * 'cone': the cone_depth rows ahead, widening by one seat on each side per
            row (Miller and Page's vision cone).
          * 'row_ahead': only the 3 seats directly in front.
          * 'moore': the 8 surrounding seats, including people behind.
        With 'cone' or 'row_ahead', the front row sees nobody and only follows its own
        judgment. Defaults to 'cone'.
    cone_depth : int, optional
        How many rows ahead an agent can see when neighborhood is 'cone'. Defaults to 3.
    number_of_pioneers : int, optional
        The number of agents who stand no matter what. Defaults to 0.
    pioneer_placement : str, optional
        Where pioneers sit: 'front', 'middle', or 'back' rows (filled from the center
        seat of each row outward), or 'random'. Defaults to 'front'.
    max_steps : int, optional
        The maximum number of peer pressure steps before stopping. Defaults to 50.
    ovation_level : float, optional
        The final share standing (greater than 0, up to 1) that counts as a standing
        ovation in the summary table. Defaults to 0.8.
    random_seed : int, optional
        The seed for the random number generator to ensure replicability. Defaults to 412.
    print_step_by_step : bool, optional
        Whether to print a text view of the hall at every step for each quality
        (. seated, S own judgment, p peer pressure, * pioneer). Best for small halls.
        Defaults to False.
    include_quality_sweep : bool, optional
        Whether to also run the model across a fine range of qualities (over several
        seeds) and add a panel showing the final share standing versus quality,
        compared with the share who stood on their own judgment. Defaults to False.
    sweep_qualities : sequence of float, optional
        The qualities used in the sweep. If None, uses 21 evenly spaced values from
        quality_threshold - 0.3 to quality_threshold + 0.2. Defaults to None.
    sweep_number_of_seeds : int, optional
        The number of random audiences averaged at each sweep quality. Defaults to 10.
    return_format : str, optional
        The format of the returned data: 'dataframe' (summary table only) or 'dict'
        (summary table plus every step of every hall and sweep results).
        Defaults to 'dataframe'.
    plot_simulation_results : bool, optional
        Whether to display the combined figure of halls and charts. Defaults to True.
    state_colors : sequence of str, optional
        Four hex color codes for seated agents, agents who stood on their own judgment,
        agents who stood from peer pressure, and pioneers. If None, uses
        ("#E6E6E6", "#3F7FBF", "#E69F00", "#D55E00"). Defaults to None.
    show_cell_borders : bool, optional
        Whether to draw faint lines between seats. Defaults to True.
    figure_size : tuple, optional
        The size of the figure in inches (width, height). If None, it is sized
        automatically from the number of qualities. Defaults to None.
    title_for_plot : str, optional
        The main title for the figure. Defaults to "Standing Ovation Model".
    subtitle_for_plot : str, optional
        The descriptive subtitle for the figure.
        Defaults to "How private judgment and peer pressure combine to decide whether an audience stands".
    caption_for_plot : str, optional
        Optional caption text displayed at the bottom of the figure. Defaults to None.
    data_source_for_plot : str, optional
        Optional data source identification text. Defaults to None.

    Returns
    -------
    pd.DataFrame or dict
        If return_format is 'dataframe', a summary table with one row per quality and
        the columns 'Quality', 'Converged', 'Steps', 'Initial Share Standing',
        'Final Share Standing', 'Share Stood On Own Judgment',
        'Share Stood From Peer Pressure', 'Share Pioneers', and 'Ovation'.
        'Initial Share Standing' is the share standing before any peer pressure.
        The three 'Share Stood' columns break down the final hall and sum to
        'Final Share Standing'. If return_format is 'dict', a dictionary with the keys:
          * 'summary': the summary table described above.
          * 'results': a dictionary keyed by quality, each holding the 'frames' (a list
            of hall arrays, one per step, where 0 is seated, 1 stood on own judgment,
            2 stood from peer pressure, and 3 pioneer), the 'share_history' list, and
            the agents' private 'signals'.
          * 'peer_thresholds': the array of each agent's personal peer threshold.
          * 'pioneers': the boolean array marking pioneer seats.
          * 'sweep': a DataFrame of sweep results ('Quality', 'Seed',
            'Initial Share Standing', 'Final Share Standing'), or None if
            include_quality_sweep is False.

    Teaching Note
    -------------
    The standing ovation model, introduced by John Miller and Scott Page, shows that
    a collective outcome can say surprisingly little about what individuals actually
    think. Whether an audience ends up on its feet depends not only on how good the
    show was, but on how diverse the audience's perceptions were, how easily people
    give in to what they see around them, and -- crucially -- who can see whom.
    Two audiences watching the same show can end in a full ovation or polite
    applause purely because of where the early standers happened to sit.

    The model makes the asymmetry of visibility concrete. With the default vision
    cone, people in the front row cannot see anyone, so they act only on their own
    judgment, while everyone behind them can see the front. A few people standing
    in the front can pull up the whole hall; the same people standing in the back
    are invisible. This is why "claques" (planted pioneers) sit up front, and why
    in organizations the opinions of highly visible people carry outsized weight.

    The model also shows that diversity matters. When a show is slightly below the
    quality threshold, an audience that perceives it identically will stay seated.
    An audience with more varied tastes produces a few enthusiasts who stand, and
    peer pressure can cascade from them. Comparing the share who stood on their own
    judgment with the final share standing separates the signal (genuine approval)
    from the amplification (conformity).

    For analysts, the lesson is to be careful about reading consensus as evidence.
    A unanimous-looking outcome -- a crowd on its feet, a market rally, a room that
    agrees -- may reflect a small number of visible early movers and a cascade of
    imitation rather than many independent judgments. Ask who could see whom, and
    who moved first.

    Examples
    --------
    # Marketing: compare four show qualities in a 20 x 30 hall
    summary = SimulateStandingOvationModel(
        qualities=(0.4, 0.5, 0.6, 0.7),
        hall_shape=(20, 30)
    )

    # Strategic seating: a mediocre show with 15 pioneers in the front rows
    results = SimulateStandingOvationModel(
        qualities=(0.4,),
        number_of_pioneers=15,
        pioneer_placement='front',
        return_format='dict'
    )

    # Diversity: a show below the threshold, with diverse tastes and a quality sweep
    SimulateStandingOvationModel(
        qualities=(0.45, 0.55),
        noise=0.3,
        peer_spread=0.2,
        include_quality_sweep=True
    )

    # Teaching: a small hall printed step by step
    SimulateStandingOvationModel(
        qualities=(0.55,),
        hall_shape=(8, 16),
        print_step_by_step=True,
        plot_simulation_results=False
    )
    """

    # Agent states
    SEATED, OWN, PEER, PIONEER = 0, 1, 2, 3
    state_symbols = {SEATED: ".", OWN: "S", PEER: "p", PIONEER: "*"}
    state_labels = ["Seated", "Stood on own judgment", "Stood from peer pressure", "Pioneer"]

    # Ensure the qualities are valid
    qualities = [float(q) for q in np.atleast_1d(qualities)]
    if len(qualities) == 0:
        raise ValueError("Please provide at least one value in the qualities argument.")
    if len(set(qualities)) != len(qualities):
        raise ValueError("Every value in the qualities argument must be unique.")

    # Ensure the perception and peer pressure arguments are valid
    if noise < 0:
        raise ValueError("noise must be at least 0.")
    if peer_threshold < 0 or peer_threshold > 1:
        raise ValueError("peer_threshold must be between 0 and 1.")
    if peer_spread < 0:
        raise ValueError("peer_spread must be at least 0.")

    # Ensure the hall shape is valid
    hall_shape = tuple(np.atleast_1d(hall_shape))
    if len(hall_shape) != 2 or any(int(n) != n or n < 1 for n in hall_shape):
        raise ValueError("hall_shape must be (rows, seats per row), using positive whole numbers.")
    hall_shape = tuple(int(n) for n in hall_shape)
    number_of_seats = hall_shape[0] * hall_shape[1]

    # Ensure the sight line arguments are valid
    if neighborhood not in ['cone', 'row_ahead', 'moore']:
        raise ValueError("neighborhood must be one of 'cone', 'row_ahead', or 'moore'.")
    if int(cone_depth) != cone_depth or cone_depth < 1:
        raise ValueError("cone_depth must be a positive whole number.")
    cone_depth = int(cone_depth)

    # Ensure the pioneer arguments are valid
    if int(number_of_pioneers) != number_of_pioneers or number_of_pioneers < 0 or number_of_pioneers > number_of_seats:
        raise ValueError("number_of_pioneers must be a whole number between 0 and the number of seats in the hall (" + str(number_of_seats) + ").")
    number_of_pioneers = int(number_of_pioneers)
    if pioneer_placement not in ['front', 'middle', 'back', 'random']:
        raise ValueError("pioneer_placement must be one of 'front', 'middle', 'back', or 'random'.")

    # Ensure the simulation and output arguments are valid
    if int(max_steps) != max_steps or max_steps < 1:
        raise ValueError("max_steps must be a positive whole number.")
    if ovation_level <= 0 or ovation_level > 1:
        raise ValueError("ovation_level must be greater than 0 and no more than 1.")
    if return_format not in ['dataframe', 'dict']:
        raise ValueError("return_format must be either 'dataframe' or 'dict'.")
    if include_quality_sweep:
        if sweep_qualities is None:
            sweep_qualities = np.linspace(quality_threshold - 0.3, quality_threshold + 0.2, 21)
        sweep_qualities = [float(q) for q in np.atleast_1d(sweep_qualities)]
        if len(sweep_qualities) == 0:
            raise ValueError("sweep_qualities must contain at least one value.")
        if int(sweep_number_of_seeds) != sweep_number_of_seeds or sweep_number_of_seeds < 1:
            raise ValueError("sweep_number_of_seeds must be a positive whole number.")

    # Ensure the state colors are valid
    if state_colors is None:
        state_colors = ["#E6E6E6", "#3F7FBF", "#E69F00", "#D55E00"]
    elif len(state_colors) != 4:
        raise ValueError("state_colors must have four colors: seated, stood on own judgment, stood from peer pressure, and pioneer.")

    # Build the (row, seat) offsets each agent can see. Row 0 is the front, so -1 is one row closer to the stage.
    if neighborhood == 'cone':
        offsets = [(-k, c) for k in range(1, cone_depth + 1) for c in range(-k, k + 1)]
    elif neighborhood == 'row_ahead':
        offsets = [(-1, -1), (-1, 0), (-1, 1)]
    else:
        offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0)]
    pad = max(max(abs(r), abs(c)) for r, c in offsets)

    # Count the visible seats for every agent (the hall's edges don't wrap)
    padded_seats = np.pad(np.ones(hall_shape, dtype=int), pad)
    seen = np.zeros(hall_shape, dtype=int)
    for r, c in offsets:
        seen += padded_seats[pad + r:pad + r + hall_shape[0], pad + c:pad + c + hall_shape[1]]

    # Calculate the share of visible seats that are standing for every agent
    def calculate_share_standing_seen(standing):
        padded_standing = np.pad(standing.astype(int), pad)
        standing_seen = np.zeros(hall_shape, dtype=int)
        for r, c in offsets:
            standing_seen += padded_standing[pad + r:pad + r + hall_shape[0], pad + c:pad + c + hall_shape[1]]
        return np.divide(standing_seen, seen, out=np.zeros(hall_shape), where=seen > 0)

    # Choose pioneer seats, filling rows from the center seat outward
    def pick_pioneers(rng):
        mask = np.zeros(hall_shape, dtype=bool)
        if number_of_pioneers == 0:
            return mask
        rows, cols = hall_shape
        if pioneer_placement == 'random':
            order = rng.permutation(number_of_seats)
        else:
            row_order = {
                'front': np.arange(rows),
                'back': np.arange(rows)[::-1],
                'middle': np.argsort(np.abs(np.arange(rows) - (rows - 1) / 2), kind='stable'),
            }[pioneer_placement]
            col_order = np.argsort(np.abs(np.arange(cols) - (cols - 1) / 2), kind='stable')
            order = np.array([r * cols + c for r in row_order for c in col_order])
        mask.flat[order[:number_of_pioneers]] = True
        return mask

    # Draw an audience: perception errors, personal peer thresholds, and pioneer seats
    def create_audience(rng):
        errors = rng.normal(0, noise, size=hall_shape)
        thresholds = np.clip(peer_threshold + rng.uniform(-peer_spread, peer_spread, size=hall_shape), 0, 1)
        pioneers = pick_pioneers(rng)
        return errors, thresholds, pioneers

    # Print a text view of the hall, with the stage at the top
    def print_hall(state, title):
        print(title + "  ({:.0%} standing)".format((state != SEATED).mean()))
        print("  " + "=" * hall_shape[1] + "  <- stage")
        for row in state:
            print("  " + "".join(state_symbols[v] for v in row))

    # Run the model for one show quality until nobody changes or max_steps is reached
    def run_model(quality, errors, thresholds, pioneers, verbose=False):
        signals = quality + errors
        state = np.where(signals > quality_threshold, OWN, SEATED)
        state[pioneers] = PIONEER
        frames = [state.copy()]
        converged = False
        if verbose:
            print_hall(state, "Quality {:.2f}, step 0".format(quality))
        for step in range(1, int(max_steps) + 1):
            standing = state != SEATED
            share = calculate_share_standing_seen(standing)
            new_state = state.copy()
            new_state[~standing & (seen > 0) & (share > thresholds)] = PEER
            if can_sit_back_down:
                new_state[standing & ~pioneers & (seen > 0) & (1 - share > thresholds)] = SEATED
            if np.array_equal(new_state, state):
                converged = True
                break
            state = new_state
            frames.append(state.copy())
            if verbose:
                print_hall(state, "Quality {:.2f}, step {}".format(quality, step))
        return {
            "frames": frames,
            "share_history": [float((f != SEATED).mean()) for f in frames],
            "signals": signals,
            "converged": converged,
            "steps": len(frames) - 1,
        }

    # Create a shared audience so that every quality is judged by the same people
    rng = np.random.default_rng(random_seed)
    errors, thresholds, pioneers = create_audience(rng)

    # Run the model at each quality
    dict_results = {}
    for quality in qualities:
        dict_results[quality] = run_model(quality, errors, thresholds, pioneers, verbose=print_step_by_step)
        if print_step_by_step:
            print()

    # Summarize the results
    final_halls = [dict_results[q]["frames"][-1] for q in qualities]
    df_summary = pd.DataFrame({
        "Quality": qualities,
        "Converged": [dict_results[q]["converged"] for q in qualities],
        "Steps": [dict_results[q]["steps"] for q in qualities],
        "Initial Share Standing": [dict_results[q]["share_history"][0] for q in qualities],
        "Final Share Standing": [dict_results[q]["share_history"][-1] for q in qualities],
        "Share Stood On Own Judgment": [float((hall == OWN).mean()) for hall in final_halls],
        "Share Stood From Peer Pressure": [float((hall == PEER).mean()) for hall in final_halls],
        "Share Pioneers": [float((hall == PIONEER).mean()) for hall in final_halls],
    })
    df_summary["Ovation"] = df_summary["Final Share Standing"] >= ovation_level

    # Run the quality sweep if requested
    df_sweep = None
    if include_quality_sweep:
        list_sweep_rows = []
        for seed_index in range(int(sweep_number_of_seeds)):
            seed = random_seed + seed_index
            sweep_errors, sweep_thresholds, sweep_pioneers = create_audience(np.random.default_rng(seed))
            for quality in sweep_qualities:
                sweep_result = run_model(quality, sweep_errors, sweep_thresholds, sweep_pioneers)
                list_sweep_rows.append({
                    "Quality": quality,
                    "Seed": seed,
                    "Initial Share Standing": sweep_result["share_history"][0],
                    "Final Share Standing": sweep_result["share_history"][-1],
                })
        df_sweep = pd.DataFrame(list_sweep_rows)

    # Generate plot if user requests it
    if plot_simulation_results:
        number_of_columns = len(qualities)
        number_of_bottom_panels = 3 if include_quality_sweep else 2

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

        # Size the figure from the number of qualities and the hall's aspect ratio
        if figure_size is None:
            figure_width = max(10, 2.6 * number_of_columns)
            # Width of each hall panel after side margins (0.92) and spacing between panels (0.2)
            cell_width = 0.92 * figure_width / (number_of_columns + 0.2 * (number_of_columns - 1))
            hall_row_height = min(cell_width * hall_shape[0] / hall_shape[1], 4)
            figure_size = (figure_width, header_height + 2 * hall_row_height + 4.3 + footer_height)
        else:
            hall_row_height = 2.4
        figure_height = figure_size[1]

        # Create figure and layout: halls after own judgment, final halls, and charts
        fig = plt.figure(figsize=figure_size)
        outer_layout = fig.add_gridspec(
            nrows=3,
            ncols=number_of_columns,
            height_ratios=[hall_row_height, hall_row_height, 3],
            hspace=0.65 / ((2 * hall_row_height + 3) / 3),
            top=1 - header_height / figure_height,
            bottom=footer_height / figure_height,
            wspace=0.2,
            left=0.06,
            right=0.98
        )
        color_map = ListedColormap(list(state_colors))

        # Draw the step 0 and final halls for each quality
        for column, quality in enumerate(qualities):
            result = dict_results[quality]
            for row, (frame_index, row_label) in enumerate(((0, "Own judgment"), (-1, "Final"))):
                ax = fig.add_subplot(outer_layout[row, column])
                ax.imshow(
                    result["frames"][frame_index],
                    cmap=color_map,
                    vmin=0,
                    vmax=3,
                    interpolation='nearest'
                )
                # Add faint lines between seats
                if show_cell_borders:
                    ax.set_xticks(np.arange(-0.5, hall_shape[1], 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, hall_shape[0], 1), minor=True)
                    ax.grid(which='minor', color="#FFFFFF", linewidth=0.3)
                ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
                for spine in ax.spines.values():
                    spine.set_color("#D9D9D9")
                    spine.set_linewidth(0.5)
                # Mark the stage with a heavy top edge
                ax.spines['top'].set_color("#262626")
                ax.spines['top'].set_linewidth(2.5)
                # Label the column with its quality, and the final hall with its outcome
                if frame_index == 0:
                    ax.set_title(
                        "Quality {:.2f}".format(quality),
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
                        outcome + "\nStanding {:.0%} → {:.0%}".format(
                            result["share_history"][0],
                            result["share_history"][-1]
                        ),
                        fontname="Arial",
                        fontsize=8,
                        color="#666666",
                        loc='left'
                    )
                # Label the rows on the first column
                if column == 0:
                    ax.set_ylabel(
                        row_label,
                        fontname="Arial",
                        fontsize=10,
                        color="#666666"
                    )

        # Create the bottom row of charts
        bottom_layout = outer_layout[2, :].subgridspec(1, number_of_bottom_panels, wspace=0.3)
        line_colors = plt.cm.viridis(np.linspace(0, 0.9, number_of_columns))

        # Plot the share standing over time for each quality
        ax_history = fig.add_subplot(bottom_layout[0, 0])
        for line_color, quality in zip(line_colors, qualities):
            ax_history.plot(
                dict_results[quality]["share_history"],
                color=line_color,
                linewidth=1.5,
                marker="o",
                markersize=3,
                label="{:.2f}".format(quality)
            )
        ax_history.axhline(
            y=ovation_level,
            color="#262626",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )
        ax_history.set_ylim(0, 1.05)
        ax_history.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
        ax_history.xaxis.get_major_locator().set_params(integer=True)
        ax_history.legend(
            title="Quality",
            title_fontsize=8,
            fontsize=8,
            frameon=False,
            loc='lower right',
            ncol=min(number_of_columns, 4),
            columnspacing=1,
            handlelength=1.5
        )

        # Plot how the final crowd stood for each quality
        ax_composition = fig.add_subplot(bottom_layout[0, 1])
        bar_positions = np.arange(number_of_columns)
        bar_left = np.zeros(number_of_columns)
        for state_column, state_color in zip(
                ["Share Stood On Own Judgment", "Share Stood From Peer Pressure", "Share Pioneers"],
                state_colors[1:]):
            ax_composition.barh(
                bar_positions,
                df_summary[state_column],
                left=bar_left,
                color=state_color,
                height=0.6
            )
            bar_left += df_summary[state_column].to_numpy()
        ax_composition.axvline(
            x=ovation_level,
            color="#262626",
            linestyle="--",
            linewidth=1,
            alpha=0.5
        )
        ax_composition.set_yticks(bar_positions)
        ax_composition.set_yticklabels(["{:.2f}".format(q) for q in qualities])
        ax_composition.invert_yaxis()
        ax_composition.set_xlim(0, 1)
        ax_composition.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))
        list_bottom_axes = [
            (ax_history, "Share standing over time", "Step", "Share standing"),
            (ax_composition, "Why the final crowd is standing", "Share of audience", "Quality"),
        ]

        # Add the quality sweep panel if requested
        if include_quality_sweep:
            ax_sweep = fig.add_subplot(bottom_layout[0, 2])
            df_sweep_summary = df_sweep.groupby("Quality")[["Initial Share Standing", "Final Share Standing"]].mean().reset_index()
            df_sweep_range = df_sweep.groupby("Quality")["Final Share Standing"].agg(['min', 'max']).reset_index()
            ax_sweep.fill_between(
                df_sweep_range["Quality"],
                df_sweep_range["min"],
                df_sweep_range["max"],
                color="#262626",
                alpha=0.12,
                linewidth=0
            )
            ax_sweep.plot(
                df_sweep_summary["Quality"],
                df_sweep_summary["Final Share Standing"],
                color="#262626",
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="With peer pressure"
            )
            # Show the share who stood on their own judgment for reference
            ax_sweep.plot(
                df_sweep_summary["Quality"],
                df_sweep_summary["Initial Share Standing"],
                color="#262626",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
                label="Own judgment only"
            )
            ax_sweep.axvline(
                x=quality_threshold,
                color="#262626",
                linewidth=0.8,
                alpha=0.3
            )
            ax_sweep.set_ylim(0, 1.05)
            ax_sweep.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
            ax_sweep.legend(fontsize=8, frameon=False, loc='upper left')
            list_bottom_axes.append(
                (ax_sweep, "Final share standing by quality", "Quality", "Share standing")
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

        # Add a legend for the agent states below the subtitle
        legend_handles = [Patch(facecolor=color, edgecolor="#D9D9D9", label=label)
                          for label, color in zip(state_labels, state_colors)]
        legend_handles.append(Patch(facecolor="#262626", edgecolor="#262626", label="Stage (top edge of each hall)"))
        fig.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.055, 1 - 0.7 / figure_height),
            ncol=len(legend_handles),
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
            "peer_thresholds": thresholds,
            "pioneers": pioneers,
            "sweep": df_sweep,
        }
