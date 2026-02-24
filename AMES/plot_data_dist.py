import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load spreadsheet
# --------------------------------------------------
file_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv"  # <-- UPDATE THIS
df = pd.read_csv(file_path)

strains = ["TA98", "TA100", "TA102", "TA1535", "TA1537", "Overall"]
labels = [-1, 0, 1]
partitions = ["Train", "Internal", "External"]

# --------------------------------------------------
# Loop over partitions
# --------------------------------------------------
for partition in partitions:
    part_df = df[df["Partition"] == partition]
    n_total = len(part_df)

    counts = []

    for strain in strains:
        for label in labels:
            n = (part_df[strain] == label).sum()
            counts.append({
                "Partition": partition,
                "Strain": strain,
                "Label": label,
                "Count": n
            })

    counts_df = pd.DataFrame(counts)

    # --------------------------------------------------
    # Print label counts
    # --------------------------------------------------
    print(f"\n📊 Label counts for partition: {partition}")
    print(counts_df.pivot(index="Strain", columns="Label", values="Count"))

    # --------------------------------------------------
    # Coverage calculations
    # --------------------------------------------------
    print(f"\n📈 Coverage for partition: {partition}")
    print(f"Total molecules: {n_total}")

    # Per-strain coverage
    for strain in strains:
        covered = part_df[strain].isin([0, 1]).sum()
        coverage_pct = 100 * covered / n_total
        print(f"{strain}: {covered}/{n_total} ({coverage_pct:.2f}%)")

    # Overall coverage (at least one non -1 label)
    overall_covered = (part_df[strains].isin([0, 1]).any(axis=1)).sum()
    overall_coverage_pct = 100 * overall_covered / n_total
    print(f"Overall coverage: {overall_covered}/{n_total} ({overall_coverage_pct:.2f}%)")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=counts_df,
        x="Strain",
        y="Count",
        hue="Label"
    )

    #plt.title(f"Label Distribution by Strain ({partition.capitalize()} Set)")
    plt.ylabel("Number of Molecules", fontsize=20)
    plt.xlabel("Strain", fontsize=20)
    plt.legend(title="Label",fontsize=16, title_fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()


    # --------------------------------------------------
    # Save high-resolution PDF
    # --------------------------------------------------
    output_name = f"label_distribution_{partition}.png"
    plt.savefig(output_name, dpi=300)
    plt.show()
