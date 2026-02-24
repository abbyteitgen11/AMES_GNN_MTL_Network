import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

df = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output/final_results_test/overlap_plotting.xlsx')
df = df.sort_values('Average', ascending=True)

mean_overlap = df['Average']
labels = df['Alert']
overlap_scores = df[['Strain 1', 'Strain 2', 'Strain 3', 'Strain 4', 'Strain 5']]

fig, ax = plt.subplots(figsize=(7, 30))

y = np.arange(len(mean_overlap))
barh_kwargs = dict(height=0.25, edgecolor="black")

ax.barh(labels, mean_overlap * 100, color="#fd8d3c", label="Mean Overlap (Toxic)", **barh_kwargs)
ax.set_xlim(0, 100)
#ax.set_yticks(y)
#ax.set_yticklabels(mean_overlap.index)
#ax.invert_yaxis()
#ax.set_xlabel("Percent Overlap")
#ax.set_title("Structural Alert Detection Performance (Overlap > 0 Definition)")
plt.tight_layout()

outpath = os.path.join("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output", "alert_performance_bars.pdf")
plt.savefig(outpath, dpi=600, transparent=True)
plt.close()



plt.figure(figsize=(8,30))
sns.heatmap(
    overlap_scores * 100,  # convert to percentage for readability
    cmap="YlOrRd",
    cbar_kws={"label": "Percent Overlap"},
    annot_kws={"size": 20},
    linewidths=0.5,
    linecolor="lightgray",
    annot=True,
    fmt=".1f"
)


#plt.title("Mean Overlap Score (Toxic Molecules Only) per Strain")
#plt.xlabel("Strain")
#plt.ylabel("Structural Alert")
plt.tight_layout()

outpath = os.path.join("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/output/", "toxic_overlap_by_strain_heatmap.pdf")
plt.savefig(outpath, dpi=600, transparent=True)  # high-res + transparent bg
plt.close()
