import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df1 = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/loss_train.xlsx')
df2 = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/loss_val.xlsx')

# Plot the data
plt.figure(figsize=(15, 10))
plt.plot(df1["Step"], df1["MTL_baseline"], marker='o', linestyle='-',label = "MTL baseline")
plt.plot(df1["Step"], df1["GNN_MTL_baseline"], marker='o', linestyle='-', label = "GNN MTL baseline")
plt.plot(df1["Step"], df1["GNN_MTL_add_pool"], marker='o', linestyle='-', label = "GNN MTL add pool")
plt.plot(df1["Step"], df1["GNN_MTL_Tanh"], marker='o', linestyle='-', label = "GNN MTL Tanh")
plt.plot(df1["Step"], df1["GNN_MTL_Relu_Tanh"], marker='o', linestyle='-', label = "GNN MTL ReLU Tanh")
plt.plot(df1["Step"], df1["GNN_MTL_nGC_3"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 3")
plt.plot(df1["Step"], df1["GNN_MTL_nGC_4"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 4")
plt.plot(df1["Step"], df1["GNN_MTL_nGC_5"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 5")
plt.plot(df1["Step"], df1["GNN_MTL_nGC_6"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 6")
plt.plot(df1["Step"], df1["GNN_MTL_nGC_1"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 1")
plt.plot(df1["Step"], df1["GNN_MTL_add_pool_Relu_nGC_5"], marker='o', linestyle='-', label = "GNN MTL add pool ReLU nGC layers = 5")

plt.xlabel("Epoch",fontsize=18)
plt.ylabel("Loss",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
#plt.legend("MTL baseline", "GNN MTL baseline", "GNN MTL add pool", "GNN MTL Tanh", "GNN MTL ReLU + Tanh", "GNN MTL nGC layers = 3", "GNN MTL nGC layers = 4", "GNN MTL nGC layers = 5", "GNN MTL nGC layers = 6", "GNN MTL nGC layers = 1", "GNN MTL add pool ReLU nGC layers = 1")
#plt.grid()
plt.show()


plt.figure(figsize=(15, 10))
plt.plot(df2["Step"], df2["MTL_baseline"], marker='o', linestyle='-', label = "MTL baseline")
plt.plot(df2["Step"], df2["GNN_MTL_baseline"], marker='o', linestyle='-', label = "GNN MTL baseline")
plt.plot(df2["Step"], df2["GNN_MTL_add_pool"], marker='o', linestyle='-', label = "GNN MTL add pool")
plt.plot(df2["Step"], df2["GNN_MTL_Tanh"], marker='o', linestyle='-', label = "GNN MTL Tanh")
plt.plot(df2["Step"], df2["GNN_MTL_Relu_Tanh"], marker='o', linestyle='-', label = "GNN MTL ReLU Tanh")
plt.plot(df2["Step"], df2["GNN_MTL_nGC_3"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 3")
plt.plot(df2["Step"], df2["GNN_MTL_nGC_4"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 4")
plt.plot(df2["Step"], df2["GNN_MTL_nGC_5"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 5")
plt.plot(df2["Step"], df2["GNN_MTL_nGC_6"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 6")
plt.plot(df2["Step"], df2["GNN_MTL_nGC_1"], marker='o', linestyle='-', label = "GNN MTL nGC layers = 1")
plt.plot(df2["Step"], df2["GNN_MTL_add_pool_Relu_nGC_5"], marker='o', linestyle='-', label = "GNN MTL add pool ReLU nGC layers = 5")

plt.xlabel("Epoch",fontsize=18)
plt.ylabel("Loss",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
#plt.legend("MTL baseline", "GNN MTL baseline", "GNN MTL add pool", "GNN MTL Tanh", "GNN MTL ReLU + Tanh", "GNN MTL nGC layers = 3", "GNN MTL nGC layers = 4", "GNN MTL nGC layers = 5", "GNN MTL nGC layers = 6", "GNN MTL nGC layers = 1", "GNN MTL add pool ReLU nGC layers = 1")
#plt.grid()
plt.show()
