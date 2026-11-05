# Probe Training & Validation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .config import SEED, MODEL_KEY, FIG_DIR


def train_layer_probes(X, y, test_size=0.25, seed=SEED):
    n_layers = X.shape[1]
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    results = {}
    for layer in range(n_layers):
        Xl = X[:, layer, :]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xl[idx_train], y[idx_train])
        acc = accuracy_score(y[idx_test], clf.predict(Xl[idx_test]))
        results[layer] = {"clf": clf, "held_out_acc": acc}
    return results, idx_train, idx_test


def shuffled_label_control(X, y, layer, n_repeats=5, seed=SEED):
    Xl = X[:, layer, :]
    accs = []
    rng = np.random.RandomState(seed)
    for i in range(n_repeats):
        y_shuffled = rng.permutation(y)
        idx_tr, idx_te = train_test_split(
            np.arange(len(y_shuffled)), test_size=0.25, random_state=seed + i, stratify=y_shuffled
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xl[idx_tr], y_shuffled[idx_tr])
        accs.append(accuracy_score(y_shuffled[idx_te], clf.predict(Xl[idx_te])))
    return np.mean(accs), np.std(accs)


def run_probe_validation(probe_X, probe_y):
    layer_results, idx_train, idx_test = train_layer_probes(probe_X, probe_y)
    layer_acc_df = pd.DataFrame({
        "layer": list(layer_results.keys()),
        "held_out_acc": [layer_results[l]["held_out_acc"] for l in layer_results],
    })
    best_layer = int(layer_acc_df.loc[layer_acc_df["held_out_acc"].idxmax(), "layer"])
    print(layer_acc_df)
    print(f"\nBest layer: {best_layer} (acc={layer_acc_df['held_out_acc'].max():.3f})")

    shuf_mean, shuf_std = shuffled_label_control(probe_X, probe_y, best_layer)
    print(f"Shuffled-label control at layer {best_layer}: {shuf_mean:.3f} +/- {shuf_std:.3f} (chance ~= 0.50)")

    real_acc = layer_acc_df.loc[layer_acc_df["layer"] == best_layer, "held_out_acc"].values[0]
    PROBE_OK = real_acc > shuf_mean + 0.15
    print(f"Real held-out acc: {real_acc:.3f} | Shuffled: {shuf_mean:.3f} | PROBE VALIDATED: {PROBE_OK}")
    assert PROBE_OK, "Probe failed validation gate -- real accuracy doesn't clearly beat the shuffled control."

    best_probe = layer_results[best_layer]["clf"]
    return layer_results, layer_acc_df, best_layer, best_probe, real_acc, shuf_mean


def plot_probe_accuracy(layer_acc_df, shuf_mean, best_layer):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_acc_df["layer"], layer_acc_df["held_out_acc"], marker="o", label="Real labels")
    ax.axhline(shuf_mean, color="gray", linestyle="--", label=f"Shuffled control ({shuf_mean:.2f})")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="Chance")
    ax.axvline(best_layer, color="green", linestyle="--", alpha=0.6, label=f"Best layer ({best_layer})")
    ax.set_xlabel("Layer"); ax.set_ylabel("Held-out accuracy")
    ax.set_title(f"Truth-probe accuracy by layer -- {MODEL_KEY}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_probe_accuracy.png"), dpi=200)
    plt.show()
