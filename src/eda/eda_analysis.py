# eda_analytical.py
import os
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

# ====================================================
#                  CONFIGURATION
# ====================================================

DATA_PATH = "../../data/cleaned/student_depression_cleaned.csv"

PLOT_DIRS = {
    "numeric": "../outputs/numeric",
    "categorical": "../outputs/categorical",
    "heatmap": "../outputs/heatmap",
    "target": "../outputs/target"
}

STATS_OUTPUT_PATH = "../outputs/statistical_tests/statistical_tests.csv"
os.makedirs(os.path.dirname(STATS_OUTPUT_PATH), exist_ok=True)

# ====================================================
#                  LOAD DATA
# ====================================================

def load_data(path=DATA_PATH):
    """Load the cleaned dataset."""
    return pd.read_csv(path)

# ====================================================
#          DETECT COLUMN TYPES
# ====================================================

def detect_column_types(df, target="Depression"):
    """Identify numeric and categorical columns, excluding target."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target in categorical_cols:
        categorical_cols.remove(target)

    return numeric_cols, categorical_cols

# ====================================================
#           NUMERIC ANALYSIS
# ====================================================

def numeric_summary(df, numeric_cols):
    """Compute descriptive statistics for numeric columns."""
    summary = df[numeric_cols].describe().T
    summary["missing"] = df[numeric_cols].isnull().sum()
    summary["unique"] = df[numeric_cols].nunique()
    return summary

# Grouped numeric summary by target
def numeric_grouped_summary(df, numeric_cols, target):
    group_summary = df.groupby(target)[numeric_cols].agg(["mean", "std", "median", "min", "max", "count"])
    return group_summary

# ====================================================
#           CATEGORICAL ANALYSIS
# ====================================================

def categorical_summary(df, categorical_cols):
    """Value counts for each categorical column."""
    cat_summary = {}
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        cat_summary[col] = pd.DataFrame({
            "Count": counts,
            "Percentage": percentages.round(2)
        })
    return cat_summary

# Grouped categorical by target
def categorical_grouped_summary(df, categorical_cols, target):
    results = {}
    for col in categorical_cols:
        results[col] = df.groupby(target)[col].value_counts(normalize=True).rename("Percentage").mul(100)
    return results

# ====================================================
#         STATISTICAL TESTS (T-test + Mann-Whitney)
# ====================================================

def run_group_stat_tests(df, numeric_col, group_col):
    """
    Perform independent samples t-test and Mann-Whitney U test.
    Assumes the grouping column is binary.
    """
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return None  # Skip non-binary groups

    g1, g2 = groups[0], groups[1]

    data1 = df[df[group_col] == g1][numeric_col].dropna()
    data2 = df[df[group_col] == g2][numeric_col].dropna()

    # Independent t-test
    t_stat, t_p = ttest_ind(data1, data2, equal_var=False)

    # Mann-Whitney U
    mw_stat, mw_p = mannwhitneyu(data1, data2, alternative="two-sided")

    return {
        "numeric_var": numeric_col,
        "group_var": group_col,
        "group_1": g1,
        "group_2": g2,
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "mannwhitney_statistic": mw_stat,
        "mannwhitney_p_value": mw_p
    }

def run_all_stat_tests(df, numeric_cols, target):
    """Run statistical tests for all numeric columns vs the target."""
    results = []
    for col in numeric_cols:
        test_result = run_group_stat_tests(df, col, target)
        if test_result:
            results.append(test_result)

    stats_df = pd.DataFrame(results)
    stats_df.to_csv(STATS_OUTPUT_PATH, index=False)
    return stats_df

# ====================================================
#           TARGET DISTRIBUTION
# ====================================================

def target_distribution(df, target="Depression"):
    counts = df[target].value_counts()
    percentages = df[target].value_counts(normalize=True) * 100
    return pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

# ====================================================
#             PRINT ANALYTICAL REPORT
# ====================================================

def print_report(df, target="Depression"):
    numeric_cols, categorical_cols = detect_column_types(df, target)

    print("\n================= ANALYTICAL REPORT =================\n")

    # Overview
    print("Dataset shape:", df.shape)
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)
    print("Target column:", target)

    # Target distribution
    print("\n--- Target Distribution ---")
    print(target_distribution(df, target))

    # Numeric analysis
    print("\n--- Numeric Summary ---")
    print(numeric_summary(df, numeric_cols))

    print("\n--- Numeric Summary Grouped by Target ---")
    print(numeric_grouped_summary(df, numeric_cols, target))

    # Categorical analysis
    print("\n--- Categorical Summary ---")
    cat_summary = categorical_summary(df, categorical_cols)
    for col, table in cat_summary.items():
        print(f"\n{col}:")
        print(table)

    print("\n--- Categorical Summary Grouped by Target ---")
    cat_grouped = categorical_grouped_summary(df, categorical_cols, target)
    for col, series in cat_grouped.items():
        print(f"\n{col}:")
        print(series)

    # Statistical tests
    print("\n--- Statistical Tests (t-test + Mann-Whitney) ---")
    stats_df = run_all_stat_tests(df, numeric_cols, target)
    print(stats_df)
    print(f"\nSaved statistical test results to: {STATS_OUTPUT_PATH}")

    # Plot references
    print("\n--- Plot References ---")
    for col in numeric_cols:
        print(f"- {col}: {os.path.join(PLOT_DIRS['numeric'], f'{col}_numeric_plots.png')}")
    for col in categorical_cols:
        print(f"- {col}: {os.path.join(PLOT_DIRS['categorical'], f'{col}_categorical_plot.png')}")

    print(f"\nCorrelation heatmap: {os.path.join(PLOT_DIRS['heatmap'], 'correlation_heatmap.png')}")
    print("\n=====================================================")

# ====================================================
#                  MAIN EXECUTION
# ====================================================

if __name__ == "__main__":
    df = load_data()
    print_report(df)
