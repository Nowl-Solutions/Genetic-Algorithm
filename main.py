
import numpy as np
import pandas as pd


# ==========================================
# DATA PREPARATION
# ==========================================

FEATURES = [
    "gender",
    "department",
    "age",
    "height",
    "weight",
    "bmi"
]


def prepare_data(df):
    df = df.copy()

    required = [
        "gender", "department",
        "age", "height", "weight"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for column in ["age", "height", "weight"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # BMI = weight (kg) / height (m)^2
    height_m = df["height"] / 100

    df["bmi"] = np.where(
        (height_m > 0) & (df["weight"] > 0),
        df["weight"] / height_m.pow(2),
        np.nan
    )

    df["bmi"] = df["bmi"].replace(
        [np.inf, -np.inf], np.nan
    )

    return df


# ==========================================
# PRECOMPUTE DISTRIBUTIONS
# ==========================================

def prepare_features(df, n_bins=5):
    encoded = {}

    for column in FEATURES:

        if column in ["gender", "department"]:
            categories = (
                df[column]
                .fillna("Missing")
                .astype(str)
            )

        else:
            values = pd.to_numeric(
                df[column], errors="coerce"
            )

            valid = values.dropna()

            if valid.empty:
                categories = pd.Series(
                    ["Missing"] * len(df),
                    index=df.index
                )

            elif valid.nunique() == 1:
                categories = values.apply(
                    lambda x: "Missing"
                    if pd.isna(x)
                    else "Same value"
                )

            else:
                _, edges = pd.qcut(
                    valid,
                    q=min(n_bins, valid.nunique()),
                    retbins=True,
                    duplicates="drop"
                )

                edges[0] = -np.inf
                edges[-1] = np.inf

                categories = (
                    pd.cut(
                        values,
                        bins=edges,
                        include_lowest=True
                    )
                    .astype(str)
                )

        codes, uniques = pd.factorize(
            categories, sort=True
        )

        encoded[column] = (
            codes,
            len(uniques)
        )

    return encoded


# ==========================================
# INITIAL POPULATION
# ==========================================

def generate_population(
    df, population_size=10, seed=None
):
    rng = np.random.default_rng(seed)
    n = len(df)

    population = []

    for _ in range(population_size):
        chromosome = np.array(
            [0] * (n // 2) +
            [1] * (n - n // 2),
            dtype=np.int8
        )

        rng.shuffle(chromosome)
        population.append(chromosome)

    return np.array(population)


# ==========================================
# FITNESS - LEXICOGRAPHIC MINIMAX
# ==========================================

def fitness(chromosome, df, encoded=None):

    if encoded is None:
        encoded = prepare_features(df)

    team_a = chromosome == 0
    team_b = chromosome == 1

    count_a = np.sum(team_a)
    count_b = np.sum(team_b)

    # Hard constraint
    if abs(count_a - count_b) > 1:
        return (0.0,) * len(FEATURES)

    if count_a == 0 or count_b == 0:
        return (0.0,) * len(FEATURES)

    scores = []

    for column in FEATURES:
        codes, n_categories = encoded[column]

        counts_a = np.bincount(
            codes[team_a],
            minlength=n_categories
        )

        counts_b = np.bincount(
            codes[team_b],
            minlength=n_categories
        )

        proportions_a = counts_a / count_a
        proportions_b = counts_b / count_b

        difference = np.sum(
            np.abs(proportions_a - proportions_b)
        )

        score = 1.0 - difference / 2.0

        scores.append(float(score))

    # Worst factor first, then second worst, ...
    return tuple(sorted(scores))


# ==========================================
# SELECTION
# ==========================================

def selection(
    population, fitness_scores, rng, k=3
):
    candidates = rng.choice(
        len(population),
        size=min(k, len(population)),
        replace=False
    )

    best_index = max(
        candidates,
        key=lambda i: tuple(fitness_scores[i])
    )

    return population[best_index].copy()


# ==========================================
# CROSSOVER
# ==========================================

def crossover(parent1, parent2, rng):

    mask = rng.random(len(parent1)) < 0.5

    child = np.where(
        mask, parent1, parent2
    ).copy()

    target_ones = np.sum(parent1 == 1)
    current_ones = np.sum(child == 1)

    if current_ones > target_ones:

        indices = np.where(child == 1)[0]

        chosen = rng.choice(
            indices,
            size=current_ones - target_ones,
            replace=False
        )

        child[chosen] = 0

    elif current_ones < target_ones:

        indices = np.where(child == 0)[0]

        chosen = rng.choice(
            indices,
            size=target_ones - current_ones,
            replace=False
        )

        child[chosen] = 1

    return child


# ==========================================
# MUTATION
# ==========================================

def mutation(
    chromosome, rng, mutation_rate=0.05
):
    child = chromosome.copy()

    if rng.random() < mutation_rate:

        team_a = np.where(child == 0)[0]
        team_b = np.where(child == 1)[0]

        if len(team_a) and len(team_b):

            a = rng.choice(team_a)
            b = rng.choice(team_b)

            child[a], child[b] = child[b], child[a]

    return child


# ==========================================
# GENETIC ALGORITHM
# ==========================================

def genetic_algorithm(
    df,
    population_size=100,
    generations=200,
    mutation_rate=0.05,
    seed=42
):

    # Ensure BMI exists
    df = prepare_data(df)

    # Precompute once, instead of every evaluation
    encoded = prepare_features(df)

    rng = np.random.default_rng(seed)

    population = generate_population(
        df, population_size, seed
    )

    history = []

    best_ever = None
    best_ever_score = None

    for gen in range(generations):

        fitness_scores = [
            fitness(chromosome, df, encoded)
            for chromosome in population
        ]

        best_index = max(
            range(len(population)),
            key=lambda i: fitness_scores[i]
        )

        best = population[best_index].copy()
        best_fitness = fitness_scores[best_index]

        if (
            best_ever_score is None
            or best_fitness > best_ever_score
        ):
            best_ever = best.copy()
            best_ever_score = best_fitness

        history.append(best_ever_score[0])

        if (gen + 1) % 10 == 0:
            print(
                f"Generation {gen + 1}/{generations}"
                f" | Worst: {best_ever_score[0]:.6f}"
                f" | Mean: {np.mean(best_ever_score):.6f}",
                flush=True
            )

        # Elitism
        new_population = [best_ever.copy()]

        while len(new_population) < population_size:

            parent1 = selection(
                population, fitness_scores, rng
            )

            parent2 = selection(
                population, fitness_scores, rng
            )

            child = crossover(
                parent1, parent2, rng
            )

            child = mutation(
                child, rng, mutation_rate
            )

            new_population.append(child)

        population = np.array(new_population)

    # Evaluate final generation
    final_scores = [
        fitness(chromosome, df, encoded)
        for chromosome in population
    ]

    final_index = max(
        range(len(population)),
        key=lambda i: final_scores[i]
    )

    if final_scores[final_index] > best_ever_score:
        best_ever = population[final_index].copy()
        best_ever_score = final_scores[final_index]

    history.append(best_ever_score[0])

    return best_ever, best_ever_score, history


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    df = pd.read_csv("HR.csv")
    df = prepare_data(df)

    print("========== DATA ==========")
    print("Employees:", len(df))
    print("Features:", FEATURES)

    print("\n========== BMI SUMMARY ==========")
    print(df["bmi"].describe())

    print("\n========== INITIAL POPULATION ==========")

    population = generate_population(
        df, population_size=10, seed=42
    )

    encoded = prepare_features(df)

    fitness_scores = [
        fitness(chromosome, df, encoded)
        for chromosome in population
    ]

    results = pd.DataFrame({
        "Chromosome": np.arange(
            1, len(population) + 1
        ),
        "Worst Score": [
            s[0] for s in fitness_scores
        ],
        "Mean Score": [
            np.mean(s) for s in fitness_scores
        ]
    })

    print(
        results.sort_values(
            ["Worst Score", "Mean Score"],
            ascending=False
        ).to_string(index=False)
    )

    print("\n========== RUN GA ==========")

    best, score, history = genetic_algorithm(
        df,
        population_size=100,
        generations=200,
        mutation_rate=0.05,
        seed=42
    )

    print("\n========== RESULT ==========")
    print("Worst Fitness:", score[0])
    print("Mean Fitness:", np.mean(score))
    print("All Fitness Scores:", score)

    team_a = df.iloc[np.where(best == 0)[0]]
    team_b = df.iloc[np.where(best == 1)[0]]

    print("Team A:", len(team_a))
    print("Team B:", len(team_b))

    print("\n========== NUMERIC MEANS ==========")

    for column in ["age", "height", "weight", "bmi"]:
        mean_a = team_a[column].mean()
        mean_b = team_b[column].mean()

        print(
            f"{column:8s}"
            f" | A: {mean_a:.3f}"
            f" | B: {mean_b:.3f}"
            f" | Difference: {abs(mean_a - mean_b):.3f}"
        )
