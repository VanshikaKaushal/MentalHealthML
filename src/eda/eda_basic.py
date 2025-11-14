# eda.py
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Disable plot windows entirely
import matplotlib.pyplot as plt
import seaborn as sns

# For nicer plots
sns.set(style="whitegrid")

# Output directory for saving plots
OUTPUT_DIR = "../outputs/eda_plots"

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================================================
#                  BASIC CHECKS
# ====================================================

def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}\n")


def unique_values_summary(df):
    print("\nUnique values per column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    print("\n")


# ====================================================
#      TARGET DISTRIBUTION (SAVED PLOT)
# ====================================================

def target_distribution(df, target='Depression'):
    counts = df[target].value_counts()
    print("\nDepression distribution:\n", counts)

    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, palette='coolwarm')
    plt.title('Depression Distribution')
    plt.xlabel('Depression (0=No, 1=Yes)')
    plt.ylabel('Count')

    plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", bbox_inches='tight')
    plt.close()


# ====================================================
#                LOAD DATA
# ====================================================

def load_data(path='../data/cleaned/student_depression_cleaned.csv'):
    df = pd.read_csv(path)
    return df


# ====================================================
#                DATA OVERVIEW
# ====================================================

def data_overview(df):
    print("\nFirst 5 rows:")
    print(df.head(), "\n")

    print("Shape of dataset:", df.shape, "\n")

    print("Column info:")
    print(df.info(), "\n")

    print("Summary statistics:")
    print(df.describe(), "\n")

    print("Missing values:")
    print(df.isnull().sum(), "\n")


# ====================================================
#         CORRELATION HEATMAP (SAVED PLOT)
# ====================================================

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    plt.figure(figsize=(10, 8))

    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Correlation Heatmap")

    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", bbox_inches='tight')
    plt.close()


# ====================================================
#      NUMERIC FEATURE DISTRIBUTIONS (SAVED PLOTS)
# ====================================================

def plot_numeric(df, numeric_cols, target='Depression'):
    for col in numeric_cols:
        plt.figure(figsize=(18, 5))

        # Histogram
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Histogram of {col}')

        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')

        # Boxplot vs target
        plt.subplot(1, 3, 3)
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f'{col} vs {target}')

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{col}_numeric_plots.png", bbox_inches='tight')
        plt.close()


# ====================================================
#     CATEGORICAL FEATURES vs TARGET (SAVED PLOTS)
# ====================================================

def plot_categorical_vs_target(df, categorical_cols, target='Depression'):
    for col in categorical_cols:
        plt.figure(figsize=(7, 4))

        sns.countplot(data=df, x=col, hue=target, palette="Set2")
        plt.title(f'{col} vs {target}')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{col}_categorical_plot.png", bbox_inches='tight')
        plt.close()


# ====================================================
#                  MAIN EXECUTION
# ====================================================

if __name__ == "__main__":
    # Load data
    df = load_data()

    # Overview
    data_overview(df)

    # Detect numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Depression' in numeric_cols:
        numeric_cols.remove('Depression')

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Generate plots
    target_distribution(df)
    plot_correlation_heatmap(df)
    plot_numeric(df, numeric_cols)
    plot_categorical_vs_target(df, categorical_cols)

    # Basic checks
    check_duplicates(df)
    unique_values_summary(df)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
