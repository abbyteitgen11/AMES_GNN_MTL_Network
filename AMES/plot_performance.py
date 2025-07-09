import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/performance/performance_full_test_GNN_MTL_layers.xlsx', index_col=0)

# Transpose so metrics are on x-axis and trials are grouped
df_T = df.transpose()

# Plot
df_T.plot(kind='bar', figsize=(14, 6))
plt.ylabel("Value")
plt.title("Performance Metrics by Model")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Model", loc='upper right')
plt.tight_layout()
plt.show()

df2 = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/performance/performance2_full_test_GNN_MTL_layers.xlsx', index_col=0)

# Transpose so metrics are on x-axis and trials are grouped
df_T2 = df2.transpose()

# Plot
df_T2.plot(kind='bar', figsize=(14, 6))
plt.ylabel("Value")
plt.title("Performance Metrics by Model")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Model", loc='upper right')
plt.tight_layout()
plt.show()

