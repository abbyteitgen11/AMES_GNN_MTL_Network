import joblib
import optuna.visualization as vis
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the study
study = joblib.load('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/optuna/study.pkl')

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

