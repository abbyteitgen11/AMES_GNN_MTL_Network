import joblib
import optuna.visualization as vis
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# SINGLE
# Load the study
study = joblib.load('/Volumes/Seagate/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/HP_opt_nonglobal/study6.pkl')

# Plot optimization history
vis.plot_optimization_history(study).show()

# Plot parameter importance
vis.plot_param_importances(study).show()


#study = optuna.load_study(study_name="my_study", storage="sqlite:///path/to/study.db")

data = []

for trial in study.trials:
    if trial.state.name == "COMPLETE":
        entry = {
            "trial_number": trial.number,
            "value": trial.value,
        }
        entry.update(trial.user_attrs)  # includes val_loss_log, best_fold, etc.
        data.append(entry)

df = pd.DataFrame(data)
print(df[["trial_number", "best_fold", "best_fold_loss"]])

# Plot val loss per epoch for one trial
trial_id = 0
val_log = study.trials[trial_id].user_attrs["val_loss_log"]

epochs = [entry["epoch"] for entry in val_log]
losses = [entry["val_loss"] for entry in val_log]

plt.plot(epochs, losses)
plt.title(f"Trial {trial_id} Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.show()

df["best_fold_val_loss"] = df["best_fold_loss"]
sns.histplot(df["best_fold_val_loss"], bins=20)
plt.title("Best Fold Validation Loss Across Trials")
plt.xlabel("Loss")
plt.show()

# Find trial with lowest best_fold_loss
best_idx = df["best_fold_loss"].idxmin()
best_trial_number = df.loc[best_idx, "trial_number"]
# Get trial object
best_trial = study.trials[best_trial_number]

# Extract parameters
best_params = best_trial.params

print("Best trial number:", best_trial_number)
print("Best fold loss:", df.loc[best_idx, "best_fold_loss"])
print("Best fold:", df.loc[best_idx, "best_fold"])
print("Best parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

"""
# Load multiple studies
study_paths = [
    '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study_5_28_25.pkl',
    '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study6.pkl'
]

all_trials = []

for path in study_paths:
    study = joblib.load(path)
    all_trials.extend([t for t in study.trials if t.state.name == "COMPLETE"])

combined_data = []

for trial in all_trials:
    entry = {
        "trial_number": trial.number,
        "value": trial.value,
        "study_source": trial.study_name if hasattr(trial, 'study_name') else path
    }
    entry.update(trial.user_attrs)
    combined_data.append(entry)

df_combined = pd.DataFrame(combined_data)

sns.histplot(df_combined["best_fold_loss"], bins=20)
plt.title("Best Fold Validation Loss Across All Studies")
plt.xlabel("Loss")
plt.show()

df_combined_sorted = df_combined.sort_values("value", ascending=False).reset_index(drop=True)
plt.plot(df_combined_sorted.index, df_combined_sorted["value"])
plt.title("Combined Optimization History")
plt.xlabel("Global Trial Number")
plt.ylabel("Objective Value")
plt.grid(True)
plt.show()

sns.boxplot(data=df_combined, x="study_source", y="best_fold_loss")
plt.title("Validation Loss by Study")
plt.show()
"""