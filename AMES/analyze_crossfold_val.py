import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Load all CSVs from a directory
csv_dir = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/MTL_publication/logits_MTL"  # Change this to your folder
csv_files = glob(os.path.join(csv_dir, "seed_*_fold_*_w_*.csv"))

pattern = re.compile(r"seed_(\d+)_fold_(\d+)_w_(\w+).csv")

# Load and tag each file
all_data = []
for file in csv_files:
    match = pattern.search(os.path.basename(file))
    if match:
        seed, fold, weight_type = match.groups()
        df = pd.read_csv(file)
        df["Seed"] = int(seed)
        df["Fold"] = int(fold)
        df["Weight"] = weight_type
        all_data.append(df)

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)

# Metrics to analyze
metrics = ["Sp", "Sn", "Prec", "Acc", "Bal acc", "F1 score", "H score"]

# Average metrics by strain and model type
avg_metrics = full_df.groupby(["Strain", "Weight"])[metrics].mean().reset_index()

# ---------- PLOT 1: Average Metrics Across All Seeds and Folds ----------
melted_avg = avg_metrics.melt(id_vars=["Strain", "Weight"], value_vars=metrics,
                              var_name="Metric", value_name="Value")

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_avg, x="Metric", y="Value", hue="Weight", ci=None)
plt.title("Average Metrics Across All Folds and Seeds by Model Type")
plt.xticks(rotation=45)
plt.legend(title="Model Type")
plt.tight_layout()
plt.show()

# ---------- PLOT 2: Boxplot of Metric Distributions Across Seeds and Folds ----------
melted_full = full_df.melt(id_vars=["Strain", "Weight", "Seed", "Fold"],
                           value_vars=metrics, var_name="Metric", value_name="Value")

plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Weight")
plt.title("Metric Distribution Across All Seeds and Folds")
plt.xticks(rotation=45)
plt.legend(title="Model Type")
plt.tight_layout()
plt.show()

# ---------- PLOT 3: Error Bars (Mean ± Std) for Metrics ----------
agg = melted_full.groupby(["Metric", "Weight"])["Value"].agg(["mean", "std"]).reset_index()

plt.figure(figsize=(12, 6))
for key, grp in agg.groupby("Weight"):
    plt.errorbar(grp["Metric"], grp["mean"], yerr=grp["std"], label=key, fmt='-o')
plt.title("Metric Means with Error Bars (Std Dev)")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title="Model Type")
plt.tight_layout()
plt.show()

# ---------- PLOT 4: Per-Seed Variability ----------
plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Seed")
plt.title("Per-Seed Variability")
plt.xticks(rotation=45)
plt.legend(title="Seed", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------- PLOT 5: Per-Fold Variability ----------
plt.figure(figsize=(14, 6))
sns.boxplot(data=melted_full, x="Metric", y="Value", hue="Fold")
plt.title("Per-Fold Variability")
plt.xticks(rotation=45)
plt.legend(title="Fold", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





"""
df = pd.read_csv('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output_seed/val_losses.csv', index_col=0)  # Use header=None if no column names
plt.figure(figsize=(8, 6))  # adjust size as needed
sns.heatmap(df, annot=True, cmap="viridis")  # annot=True shows values in cells
plt.xlabel("Seed")
plt.ylabel("Fold")
plt.tight_layout()
plt.show()
"""



