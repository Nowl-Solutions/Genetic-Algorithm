
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from main import genetic_algorithm, prepare_data


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "HR.csv"
OUTPUT_PATH = BASE_DIR / "ga_dashboard.png"

TEAM_COLORS = ["#3478F6", "#FF8A4C"]

POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.05
SEED = 42


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_team_data(df, chromosome):
    team_a = df.iloc[
        np.where(chromosome == 0)[0]
    ].copy()

    team_b = df.iloc[
        np.where(chromosome == 1)[0]
    ].copy()

    return team_a, team_b


def get_numeric_data(team, column):
    return pd.to_numeric(
        team[column], errors="coerce"
    ).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


def get_category_counts(team_a, team_b, column):
    counts = pd.DataFrame({
        "Team A": (
            team_a[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        ),
        "Team B": (
            team_b[column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )
    }).fillna(0)

    return counts.astype(int)


def format_value(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


# ==========================================
# NUMERIC DISTRIBUTION PLOT
# ==========================================

def plot_numeric_distribution(
    ax, team_a, team_b,
    column, label, unit
):
    values_a = get_numeric_data(team_a, column)
    values_b = get_numeric_data(team_b, column)

    mean_a = values_a.mean()
    mean_b = values_b.mean()

    difference = abs(mean_a - mean_b)

    combined = pd.concat(
        [values_a, values_b]
    )

    if not combined.empty:
        bins = np.histogram_bin_edges(
            combined.to_numpy(),
            bins=12
        )

        ax.hist(
            [values_a, values_b],
            bins=bins,
            color=TEAM_COLORS,
            label=["Team A", "Team B"],
            alpha=0.75,
            edgecolor="white"
        )

        if pd.notna(mean_a):
            ax.axvline(
                mean_a,
                color=TEAM_COLORS[0],
                linestyle="--",
                linewidth=2
            )

        if pd.notna(mean_b):
            ax.axvline(
                mean_b,
                color=TEAM_COLORS[1],
                linestyle="--",
                linewidth=2
            )

        ax.legend(fontsize=9)

    ax.set_title(
        f"{label} Distribution\n"
        f"Mean A: {format_value(mean_a)} | "
        f"Mean B: {format_value(mean_b)}\n"
        f"Difference: {format_value(difference)} {unit}",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("Employees")
    ax.grid(axis="y", alpha=0.2)

    return [
        f"Mean {label} ({unit})",
        format_value(mean_a),
        format_value(mean_b),
        format_value(difference)
    ]


# ==========================================
# MAIN DASHBOARD
# ==========================================

def plot_dashboard(df, best, score, history):

    # Ensure BMI exists
    df = prepare_data(df)

    team_a, team_b = get_team_data(df, best)

    worst_fitness = float(score[0])
    mean_fitness = float(np.mean(score))

    fig = plt.figure(
        figsize=(20, 19),
        layout="constrained"
    )

    gs = fig.add_gridspec(
        5, 4,
        height_ratios=[
            1.0,
            1.3,
            1.4,
            1.4,
            1.1
        ]
    )

    fig.suptitle(
        "GENETIC ALGORITHM - TEAM BALANCING\n"
        "6 FACTORS: GENDER | DEPARTMENT | AGE | "
        "HEIGHT | WEIGHT | BMI\n"
        f"Worst Fitness: {worst_fitness:.6f} | "
        f"Mean Fitness: {mean_fitness:.6f} | "
        f"Team A: {len(team_a)} | "
        f"Team B: {len(team_b)}",
        fontsize=17,
        fontweight="bold"
    )

    # ======================================
    # 1. TEAM SIZE
    # ======================================

    ax = fig.add_subplot(gs[0, 0])

    bars = ax.bar(
        ["Team A", "Team B"],
        [len(team_a), len(team_b)],
        color=TEAM_COLORS,
        width=0.55
    )

    ax.bar_label(bars, padding=4)
    ax.set_ylim(
        0,
        max(len(team_a), len(team_b)) * 1.2
    )

    ax.set_title("Team Size")
    ax.set_ylabel("Employees")
    ax.grid(axis="y", alpha=0.2)

    # ======================================
    # 2. GENDER COMPARISON
    # ======================================

    ax = fig.add_subplot(gs[0, 1:3])

    gender = get_category_counts(
        team_a, team_b, "gender"
    )

    gender.plot.bar(
        ax=ax,
        color=TEAM_COLORS,
        rot=0
    )

    ax.set_title("Gender Comparison")
    ax.set_ylabel("Employees")
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # ======================================
    # 3. FITNESS CONVERGENCE
    # ======================================

    ax = fig.add_subplot(gs[0, 3])

    ax.plot(
        range(1, len(history) + 1),
        history,
        color="#22A06B",
        linewidth=2
    )

    ax.set_title("GA Fitness Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Worst Fitness")
    ax.grid(alpha=0.3)

    # ======================================
    # 4. DEPARTMENT COMPARISON
    # ======================================

    ax = fig.add_subplot(gs[1, :])

    department = get_category_counts(
        team_a, team_b, "department"
    )

    department.plot.bar(
        ax=ax,
        color=TEAM_COLORS,
        width=0.8,
        rot=25
    )

    ax.set_title(
        "Department Distribution",
        fontweight="bold"
    )

    ax.set_ylabel("Employees")
    ax.set_xlabel("")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    # ======================================
    # 5-8. NUMERIC DISTRIBUTIONS (2x2)
    # ======================================

    numeric_columns = [
        ("age", "Age", "years"),
        ("height", "Height", "cm"),
        ("weight", "Weight", "kg"),
        ("bmi", "BMI", "kg/m²")
    ]

    positions = [
        gs[2, 0:2],
        gs[2, 2:4],
        gs[3, 0:2],
        gs[3, 2:4]
    ]

    summary_rows = [
        [
            "Employees",
            str(len(team_a)),
            str(len(team_b)),
            str(abs(len(team_a) - len(team_b)))
        ]
    ]

    for position, (column, label, unit) in zip(
        positions, numeric_columns
    ):
        ax = fig.add_subplot(position)

        row = plot_numeric_distribution(
            ax,
            team_a,
            team_b,
            column,
            label,
            unit
        )

        summary_rows.append(row)

    # ======================================
    # 9. SUMMARY TABLE
    # ======================================

    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")

    table = ax.table(
        cellText=summary_rows,
        colLabels=[
            "Metric",
            "Team A",
            "Team B",
            "Difference"
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.2, 0.2, 0.2]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():

        if row == 0:
            cell.set_facecolor("#26354A")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")

        elif row % 2 == 0:
            cell.set_facecolor("#EDF2F8")

    ax.set_title(
        "Team Summary & Mean Differences",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    # ======================================
    # SAVE AND SHOW
    # ======================================

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"\nDashboard saved: {OUTPUT_PATH}")

    print("\n========== TEAM SUMMARY ==========")

    print(
        pd.DataFrame(
            summary_rows,
            columns=[
                "Metric",
                "Team A",
                "Team B",
                "Difference"
            ]
        ).to_string(index=False)
    )

    plt.show()


# ==========================================
# RUN GENETIC ALGORITHM
# ==========================================

if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    # Calculate BMI automatically
    df = prepare_data(df)

    print("========== GENETIC ALGORITHM ==========")
    print(f"Employees: {len(df)}")
    print(f"Population: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print("Factors: 6")

    best, score, history = genetic_algorithm(
        df,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        seed=SEED
    )

    print("\n========== GA RESULT ==========")
    print(f"Worst Fitness: {score[0]:.6f}")
    print(f"Mean Fitness: {np.mean(score):.6f}")
    print(f"Team A: {np.sum(best == 0)}")
    print(f"Team B: {np.sum(best == 1)}")

    plot_dashboard(
        df=df,
        best=best,
        score=score,
        history=history
    )
