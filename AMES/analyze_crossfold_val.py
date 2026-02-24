"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from glob import glob

# --------------------------------------------------
# Load all CSVs from a directory
# --------------------------------------------------
csv_dir = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output_seed_8_20_25/metrics"  # Change this to your folder
csv_files = glob(os.path.join(csv_dir, "metrics_seed_*_fold_*.csv"))

pattern = re.compile(r"metrics_seed_(\d+)_fold_(\d+).csv")

# Load and tag each file
all_data = []
for file in csv_files:
    match = pattern.search(os.path.basename(file))
    if match:
        seed, fold = match.groups()
        df = pd.read_csv(file)
        df["Seed"] = int(seed)
        df["Fold"] = int(fold)
        all_data.append(df)

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)

# --------------------------------------------------
# Metrics to analyze
# --------------------------------------------------
metrics = ["Sp", "Sn", "Prec", "Acc", "Bal acc", "F1 score", "H score"]

# --------------------------------------------------
# Average metrics by strain (across all seeds and folds)
# --------------------------------------------------
avg_metrics = full_df.groupby(["Strain"])[metrics].mean().reset_index()

# Average metrics by seed (averaging across all 5 folds)
avg_seed = full_df.groupby(["Seed"])[metrics].mean().reset_index()

# --------------------------------------------------
# ---------- PLOT 1: Average Metrics Across All Seeds and Folds ----------
# --------------------------------------------------
melted_avg = avg_metrics.melt(id_vars=["Strain"], value_vars=metrics,
                              var_name="Metric", value_name="Value")

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_avg, x="Metric", y="Value", hue="Strain", ci=None)
plt.title("Average Metrics Across All Folds and Seeds by Strain")
plt.xticks(rotation=45)
plt.legend(title="Strain")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# ---------- PLOT 2: Boxplot of Metric Distributions Across Seeds and Folds ----------
# --------------------------------------------------
melted_full = full_df.melt(id_vars=["Strain", "Seed", "Fold"],
                           value_vars=metrics, var_name="Metric", value_name="Value")

plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Strain")
plt.title("Metric Distribution Across All Seeds and Folds")
plt.xticks(rotation=45)
plt.legend(title="Strain")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# ---------- PLOT 3: Error Bars (Mean ± Std) for Metrics ----------
# --------------------------------------------------
agg = melted_full.groupby(["Metric", "Strain"])["Value"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(12, 6))
for key, grp in agg.groupby("Strain"):
    plt.errorbar(grp["Metric"], grp["mean"], yerr=grp["std"], label=key, fmt='-o')
plt.title("Metric Means with Error Bars (Std Dev)")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title="Strain")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# ---------- PLOT 4: Per-Seed Variability ----------
# --------------------------------------------------
plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Seed")
plt.title("Per-Seed Variability")
plt.xticks(rotation=45)
plt.legend(title="Seed", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# ---------- PLOT 5: Per-Fold Variability ----------
# --------------------------------------------------
plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Fold")
plt.title("Per-Fold Variability")
plt.xticks(rotation=45)
plt.legend(title="Fold", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# ---------- TOP 5 PERFORMING SEEDS (based on mean Balanced Accuracy) ----------
# --------------------------------------------------

# Step 1: Average Balanced Accuracy across folds and strains for each seed
seed_balacc = full_df.groupby("Seed")["Bal acc"].mean().reset_index()

# Step 2: Sort by average Balanced Accuracy (descending)
top_5_seeds = seed_balacc.sort_values("Bal acc", ascending=False).head(5)
print("\n🏆 Top 5 Seeds by Average Balanced Accuracy Across All 5 Folds:")
print(top_5_seeds)

# Step 3: For each top seed, find the fold with the best Balanced Accuracy
print("\n📊 Best Fold per Top Seed:")
best_folds = []
for seed in top_5_seeds["Seed"]:
    seed_folds = full_df[full_df["Seed"] == seed]
    fold_avg = seed_folds.groupby("Fold")["Bal acc"].mean().reset_index()
    best_fold = fold_avg.loc[fold_avg["Bal acc"].idxmax()]
    best_folds.append({"Seed": seed, "Best Fold": int(best_fold["Fold"]), "Bal acc": best_fold["Bal acc"]})

best_folds_df = pd.DataFrame(best_folds)
print(best_folds_df)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from glob import glob
import numpy as np
import matplotlib
from matplotlib import rc

matplotlib.rcParams['savefig.transparent'] = True

# --------------------------------------------------
# Load all CSVs from a directory
# --------------------------------------------------
csv_dir = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output_seed_8_20_25/metrics"
csv_files = glob(os.path.join(csv_dir, "metrics_seed_*_fold_*.csv"))

pattern = re.compile(r"metrics_seed_(\d+)_fold_(\d+).csv")

all_data = []
for file in csv_files:
    match = pattern.search(os.path.basename(file))
    if match:
        seed, fold = match.groups()
        df = pd.read_csv(file)

        # --------------------------------------------------
        # Compute NPV and MCC from TP, TN, FP, FN
        # --------------------------------------------------
        TP = df["TP"]
        TN = df["TN"]
        FP = df["FP"]
        FN = df["FN"]

        df["NPV"] = TN / (TN + FN)

        df["MCC"] = (
            (TP * TN - FP * FN) /
            np.sqrt(
                (TP + FP) *
                (TP + FN) *
                (TN + FP) *
                (TN + FN)
            )
        )

        # --------------------------------------------------
        # Standardize column names used downstream
        # --------------------------------------------------
        df = df.rename(columns={
            "Strain": "Strain",
            "Sp": "Specificity",
            "Sn": "Sensitivity",
            "Prec": "PPV",
            "Acc": "Accuracy",
            "Bal acc": "Balanced Accuracy",
            "F1 score": "F1 Score",
            "H score": "H1 Score"
        })

        df["Seed"] = int(seed)
        df["Fold"] = int(fold)
        all_data.append(df)

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)

# --------------------------------------------------
# Metrics to analyze (ORDER MATTERS)
# --------------------------------------------------
metrics = [
    "Specificity",
    "Sensitivity",
    "Accuracy",
    "Balanced Accuracy",
    "PPV",
    "NPV",
    "MCC",
    "F1 Score",
    "H1 Score"
]

# --------------------------------------------------
# Average metrics by strain (across all seeds and folds)
# --------------------------------------------------
avg_metrics = full_df.groupby(["Strain"])[metrics].mean().reset_index()

# Average metrics by seed (averaging across all 5 folds)
avg_seed = full_df.groupby(["Seed"])[metrics].mean().reset_index()

# --------------------------------------------------
# ---------- PLOT 1: Average Metrics Across All Seeds and Folds ----------
# --------------------------------------------------
melted_avg = avg_metrics.melt(
    id_vars=["Strain"],
    value_vars=metrics,
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_avg, x="Metric", y="Value", hue="Strain", ci=None)
plt.title("Average Metrics Across All Folds and Seeds by Strain")
plt.xticks(rotation=45)
plt.legend(title="Strain")
plt.tight_layout()
plt.savefig("avg_metrics_by_strain.png", dpi=300)
plt.show()

# --------------------------------------------------
# ---------- PLOT 2: Boxplot of Metric Distributions ----------
# --------------------------------------------------
melted_full = full_df.melt(
    id_vars=["Strain", "Seed", "Fold"],
    value_vars=metrics,
    var_name="Metric",
    value_name="Value"
)

#sns.set(font_scale=1.4)
#sns.set(rc={'axes.edgecolor':'black'})


plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Strain")
#plt.title("Performance Distribution Across All Seeds and Folds")
plt.xticks(rotation=45)
plt.legend(title="Strain", fontsize=12, title_fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Metric",fontsize=14)
plt.ylabel("Value",fontsize=14)
plt.tight_layout()
plt.savefig("metric_distribution_by_strain.png", dpi=300)
plt.show()

# --------------------------------------------------
# ---------- PLOT 3: Error Bars (Mean ± Std) ----------
# --------------------------------------------------
agg = melted_full.groupby(["Metric", "Strain"])["Value"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(12, 6))
for key, grp in agg.groupby("Strain"):
    plt.errorbar(grp["Metric"], grp["mean"], yerr=grp["std"], label=key, fmt='-o')

plt.title("Metric Means with Error Bars (Std Dev)")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title="Strain")
plt.tight_layout()
plt.savefig("metric_error_bars.png", dpi=300)
plt.show()

# --------------------------------------------------
# ---------- PLOT 4: Per-Seed Variability ----------
# --------------------------------------------------
plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Seed")
plt.title("Per-Seed Variability")
plt.xticks(rotation=45)
plt.legend(title="Seed", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("per_seed_variability.png", dpi=300)
plt.show()

# --------------------------------------------------
# ---------- PLOT 5: Per-Fold Variability ----------
# --------------------------------------------------
plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Fold")
plt.title("Per-Fold Variability")
plt.xticks(rotation=45)
plt.legend(title="Fold", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("per_fold_variability.png", dpi=300)
plt.show()

# --------------------------------------------------
# ---------- TOP 5 PERFORMING SEEDS (Balanced Accuracy) ----------
# --------------------------------------------------
seed_balacc = full_df.groupby("Seed")["Balanced Accuracy"].mean().reset_index()

top_5_seeds = seed_balacc.sort_values("Balanced Accuracy", ascending=False).head(5)
print("\n🏆 Top 5 Seeds by Average Balanced Accuracy Across All 5 Folds:")
print(top_5_seeds)

print("\n📊 Best Fold per Top Seed:")
best_folds = []
for seed in top_5_seeds["Seed"]:
    seed_folds = full_df[full_df["Seed"] == seed]
    fold_avg = seed_folds.groupby("Fold")["Balanced Accuracy"].mean().reset_index()
    best_fold = fold_avg.loc[fold_avg["Balanced Accuracy"].idxmax()]
    best_folds.append({
        "Seed": seed,
        "Best Fold": int(best_fold["Fold"]),
        "Balanced Accuracy": best_fold["Balanced Accuracy"]
    })

best_folds_df = pd.DataFrame(best_folds)
print(best_folds_df)




# --------------------------------------------------
# Load validation loss spreadsheet
# --------------------------------------------------
file_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/validation_loss_for_figure.xlsx"
df = pd.read_excel(file_path, index_col=0)

# --------------------------------------------------
# Plot heatmap
# --------------------------------------------------
sns.set(font_scale=1.2)

plt.figure(figsize=(10, 6))
sns.heatmap(
    df,
    annot=True,
    fmt=".4f",
    cmap="viridis",
    cbar_kws={"label": "Validation Loss"},
)

#plt.title("Validation Loss Across Seeds and Folds")
plt.xlabel("Random Seed",fontsize=16)
plt.ylabel("Fold",fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# --------------------------------------------------
# Save high-resolution figure
# --------------------------------------------------
plt.savefig("validation_loss_heatmap.png", dpi=300)
plt.show()




