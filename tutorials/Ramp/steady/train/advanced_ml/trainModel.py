import argparse
import numpy as np
from mpi4py import MPI
from pyofm import PYOFM
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import os

print(tf.__version__)

np.set_printoptions(precision=6, suppress=False)
np.random.seed(0)

# =============================================================================
# CLI Arguments
# =============================================================================
parser = argparse.ArgumentParser(description="Advanced FIML training with feature selection and architecture search")
parser.add_argument("--n_features", type=int, default=6, help="Number of top features to select (default: 6)")
parser.add_argument("--skip_search", action="store_true", help="Skip architecture search, use default [20,20] tanh")
parser.add_argument("--epochs", type=int, default=500, help="Training epochs for final model (default: 500)")
parser.add_argument("--features", type=str, default=None, help="Comma-separated feature names to use (bypasses auto-selection)")
args = parser.parse_args()

# =============================================================================
# Constants
# =============================================================================
nCells = 5000
cases = ["../c1_data", "../c2_data"]
all_features = ["VoS", "PoD", "chiSA", "pGradStream", "PSoSS", "SCurv", "UOrth", "KoU2", "ReWall", "CoP", "TauoK"]

# =============================================================================
# Phase A: Data Loading & Feature Selection
# =============================================================================
print("=" * 70)
print("PHASE A: Data Loading & Feature Selection")
print("=" * 70)

ofm = PYOFM(comm=MPI.COMM_WORLD)

# Verify data directories exist
for case in cases:
    if not os.path.exists(case):
        raise FileNotFoundError(
            f"Data directory {case} not found. "
            f"Run field inversion first: mpirun -np 4 python advanced_ml/runScript_FI.py -index 0/1"
        )

# Load all features and target
inputs = None
outputs = None
for case in cases:
    input_case = []
    output_case = []

    for feature in all_features:
        field = np.zeros(nCells)
        ofm.readField(feature, "volScalarField", case, field)
        input_case.append(field)
    input_case = np.asarray(input_case).transpose()

    field = np.zeros(nCells)
    ofm.readField("betaFIOmega", "volScalarField", case, field)
    output_case.append(field)
    output_case = np.asarray(output_case).transpose()

    if inputs is None:
        inputs = np.copy(input_case)
    else:
        inputs = np.concatenate((inputs, input_case), axis=0)
    if outputs is None:
        outputs = np.copy(output_case)
    else:
        outputs = np.concatenate((outputs, output_case), axis=0)

targets = outputs[:, 0]
n_samples = inputs.shape[0]
print(f"\nLoaded {n_samples} samples with {len(all_features)} features")

# Compute feature rankings
print("\n--- Feature Ranking ---")
print(f"{'Feature':<15} {'Spearman rho':>14} {'Spearman rank':>14} {'MI score':>10} {'MI rank':>10} {'Combined':>10}")
print("-" * 75)

spearman_scores = []
for i in range(len(all_features)):
    rho, _ = spearmanr(inputs[:, i], targets)
    spearman_scores.append(abs(rho))

mi_scores = mutual_info_regression(inputs, targets, random_state=0)

# Rank features (lower rank = better)
spearman_ranks = np.argsort(np.argsort(-np.array(spearman_scores))) + 1
mi_ranks = np.argsort(np.argsort(-mi_scores)) + 1
combined_ranks = spearman_ranks + mi_ranks

for i, feat in enumerate(all_features):
    print(
        f"{feat:<15} {spearman_scores[i]:>14.4f} {spearman_ranks[i]:>14d} "
        f"{mi_scores[i]:>10.4f} {mi_ranks[i]:>10d} {combined_ranks[i]:>10d}"
    )

# Select features
if args.features is not None:
    selected_names = [f.strip() for f in args.features.split(",")]
    selected_idx = [all_features.index(f) for f in selected_names]
else:
    sorted_idx = np.argsort(combined_ranks)
    selected_idx = sorted_idx[: args.n_features].tolist()
    selected_names = [all_features[i] for i in selected_idx]

print(f"\nSelected {len(selected_names)} features: {selected_names}")

# Subset inputs to selected features
inputs_selected = inputs[:, selected_idx]

# Save feature analysis report
os.makedirs("results", exist_ok=True)
with open("results/feature_analysis.txt", "w") as f:
    f.write("Feature Selection Report\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total samples: {n_samples}\n")
    f.write(f"Total features: {len(all_features)}\n")
    f.write(f"Selected features (top {len(selected_names)}): {selected_names}\n\n")
    f.write(f"{'Feature':<15} {'|Spearman|':>12} {'MI score':>10} {'Combined rank':>14}\n")
    f.write("-" * 55 + "\n")
    for i, feat in enumerate(all_features):
        marker = " <-- selected" if i in selected_idx else ""
        f.write(f"{feat:<15} {spearman_scores[i]:>12.4f} {mi_scores[i]:>10.4f} {combined_ranks[i]:>14d}{marker}\n")

print("Feature analysis saved to results/feature_analysis.txt")

# =============================================================================
# Phase B: Architecture Search
# =============================================================================
print("\n" + "=" * 70)
print("PHASE B: Architecture Search")
print("=" * 70)

if args.skip_search:
    best_hidden = [20, 20]
    best_activation = "tanh"
    print("Skipping search, using default: [20, 20] tanh")
else:
    hidden_configs = [[20], [20, 20], [50], [50, 50], [20, 20, 20], [50, 20]]
    activations = ["relu", "tanh", "sigmoid"]

    results = []

    for hidden in hidden_configs:
        for act in activations:
            tf.random.set_seed(0)
            np.random.seed(0)

            normalizer = layers.Normalization(input_shape=[len(selected_names)], axis=None)
            normalizer.adapt(inputs_selected)

            model_layers = [normalizer]
            for n_units in hidden:
                model_layers.append(layers.Dense(units=n_units, activation=act))
            model_layers.append(layers.Dense(units=1))

            model = tf.keras.Sequential(model_layers)
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

            history = model.fit(
                inputs_selected,
                targets,
                epochs=300,
                batch_size=500,
                validation_split=0.2,
                verbose=0,
            )

            train_mse = history.history["loss"][-1]
            val_mse = history.history["val_loss"][-1]
            n_params = model.count_params()
            results.append((hidden, act, train_mse, val_mse, n_params))

            print(f"  {str(hidden):<16} {act:<10} train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  params={n_params}")

    # Sort by validation MSE
    results.sort(key=lambda x: x[3])

    print("\n--- Architecture Search Results (sorted by val MSE) ---")
    print(f"{'Hidden':<16} {'Activation':<12} {'Train MSE':>12} {'Val MSE':>12} {'Params':>8}")
    print("-" * 62)
    for hidden, act, train_mse, val_mse, n_params in results:
        print(f"{str(hidden):<16} {act:<12} {train_mse:>12.6f} {val_mse:>12.6f} {n_params:>8d}")

    best_hidden, best_activation = results[0][0], results[0][1]
    print(f"\nBest architecture: hidden={best_hidden}, activation={best_activation}")

    # Save search results
    with open("results/architecture_search.txt", "w") as f:
        f.write("Architecture Search Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Features used: {selected_names}\n")
        f.write(f"Search epochs: 300\n\n")
        f.write(f"{'Hidden':<16} {'Activation':<12} {'Train MSE':>12} {'Val MSE':>12} {'Params':>8}\n")
        f.write("-" * 62 + "\n")
        for hidden, act, train_mse, val_mse, n_params in results:
            f.write(f"{str(hidden):<16} {act:<12} {train_mse:>12.6f} {val_mse:>12.6f} {n_params:>8d}\n")
        f.write(f"\nBest: hidden={best_hidden}, activation={best_activation}\n")

# =============================================================================
# Phase C: Final Training & Export
# =============================================================================
print("\n" + "=" * 70)
print("PHASE C: Final Training & Export")
print("=" * 70)

tf.random.set_seed(0)
np.random.seed(0)

normalizer = layers.Normalization(input_shape=[len(selected_names)], axis=None)
normalizer.adapt(inputs_selected)

final_layers = [normalizer]
for n_units in best_hidden:
    final_layers.append(layers.Dense(units=n_units, activation=best_activation))
final_layers.append(layers.Dense(units=1))

model = tf.keras.Sequential(final_layers)
model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error")

history = model.fit(
    inputs_selected,
    targets,
    epochs=args.epochs,
    batch_size=500,
    validation_split=0.2,
)

# Evaluate
predictions = model.predict(inputs_selected, verbose=0)[:, 0]
mse = np.mean((targets - predictions) ** 2)
ss_res = np.sum((targets - predictions) ** 2)
ss_tot = np.sum((targets - np.mean(targets)) ** 2)
r2 = 1.0 - ss_res / ss_tot
max_error = np.max(np.abs(targets - predictions))

print(f"\nFinal Metrics:")
print(f"  MSE:       {mse:.6f}")
print(f"  R^2:       {r2:.6f}")
print(f"  Max error: {max_error:.6f}")
print(f"  Params:    {model.count_params()}")

# Save model
model.save("model")
print("Model saved to model/")

# Save predicted beta fields for debugging
outputs_out = np.array(predictions, dtype="d")
ofm.writeField("betaFIOmegaNN_C1", "volScalarField", outputs_out[:nCells])
ofm.writeField("betaFIOmegaNN_C2", "volScalarField", outputs_out[nCells:])
print("Predicted beta fields written: betaFIOmegaNN_C1, betaFIOmegaNN_C2")

# Save final summary
with open("results/training_summary.txt", "w") as f:
    f.write("Final Training Summary\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Selected features: {selected_names}\n")
    f.write(f"Architecture: hidden={best_hidden}, activation={best_activation}\n")
    f.write(f"Epochs: {args.epochs}\n")
    f.write(f"Parameters: {model.count_params()}\n\n")
    f.write(f"MSE:       {mse:.6f}\n")
    f.write(f"R^2:       {r2:.6f}\n")
    f.write(f"Max error: {max_error:.6f}\n")

print("\nDone. Results saved to results/")
