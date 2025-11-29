import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Feature importance data
df_imp = pd.DataFrame({
    "feature": [
        "Have_you_ever_had_suicidal_thoughts_",
        "Academic_Pressure",
        "Financial_Stress",
        "Dietary_Habits",
        "Age",
        "Work_Study_Hours",
        "Study_Satisfaction",
        "Sleep_Duration",
        "Family_History_of_Mental_Illness",
        "Degree_Category_HighSchool",
        "CGPA",
        "Degree_Category_Graduate",
        "Degree_Category_PhD",
        "Gender_Female",
        "Gender_Male",
        "Degree_Category_STEM_Bach",
        "Degree_Category_Business",
        "Degree_Category_Arts"
    ],
    "importance": [
        387.075928,
        118.320831,
        77.186317,
        28.149897,
        25.591812,
        22.815075,
        18.308125,
        11.655374,
        7.978407,
        5.273252,
        3.987113,
        3.515975,
        2.871779,
        2.865262,
        2.840335,
        2.701330,
        2.638389,
        2.593024
    ]
})

# Sort features by importance
df_imp = df_imp.sort_values("importance", ascending=True)

# Normalize importance for color mapping
norm = (df_imp["importance"] - df_imp["importance"].min()) / (df_imp["importance"].max() - df_imp["importance"].min())

# Pastel rainbow colormap
pastel_colors = [
    "#FFB3BA",  # Red
    "#FFDFBA",  # Orange
    "#FFFFBA",  # Yellow
    "#BAFFC9",  # Green
    "#BAE1FF",  # Blue
    "#E2BAFF"   # Purple
]
cmap = mcolors.LinearSegmentedColormap.from_list("pastel_rainbow", pastel_colors, N=256)
colors = cmap(norm)

# Plot
plt.figure(figsize=(10, 7))
bars = plt.barh(df_imp["feature"], df_imp["importance"], color=colors)

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 3, bar.get_y() + bar.get_height()/2,
             f"{width:.1f}", va='center', fontsize=9)

plt.xlabel("Importance Score", fontsize=12)
plt.title("XGBoost Feature Importance (Pastel Rainbow)", fontsize=14, weight="bold")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.grid(False)
plt.tight_layout()
plt.show()
