import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from . import model as M
from .config import MODEL_KEY, MODEL_NAME, MODEL_CANDIDATES, FIG_DIR, TABLE_DIR, SUMMARY_DIR


def layerwise_trajectory(hs, layer_probes):
    return np.array([
        layer_probes[layer]["clf"].predict_proba(hs[layer, :].reshape(1, -1))[0, 1]
        for layer in sorted(layer_probes.keys())
    ])


def run_analysis(debate_df, layer_results, layer_acc_df, best_layer, best_probe,
                 real_acc, shuf_mean, n_truth_items):
    # ---- Funnel ----
    n_total = len(debate_df)
    usable_mask = (
        (debate_df["baseline_choice"] == "correct")
        & (debate_df["debate_choice"] != "unclear")
        & (debate_df["control_choice"] != "unclear")
    )
    usable_df = debate_df[usable_mask].copy()
    n_usable = len(usable_df)
    usable_df["flipped"] = usable_df["debate_choice"] == "wrong"
    n_flipped = int(usable_df["flipped"].sum())
    print(f"Total: {n_total} | Usable: {n_usable} | Flipped: {n_flipped} | Non-flipped: {n_usable - n_flipped}")

    # Result 1: Behavioral flip rate 
    flip_rate_overall = usable_df["flipped"].mean()
    flip_rate_by_domain = usable_df.groupby("domain")["flipped"].mean()
    flip_rate_by_difficulty = usable_df.groupby("difficulty")["flipped"].mean()
    print(f"\nOverall flip rate: {flip_rate_overall:.1%}")
    print("By domain:\n", flip_rate_by_domain)
    print("By difficulty:\n", flip_rate_by_difficulty)

    # Result 2: Probe confidence, control vs debate (paired t-test)
    def probe_confidence(hs_row, layer=best_layer, probe=best_probe):
        return probe.predict_proba(hs_row[layer, :].reshape(1, -1))[0, 1]

    usable_df["debate_confidence"] = usable_df["debate_hs"].apply(probe_confidence)
    usable_df["control_confidence"] = usable_df["control_hs"].apply(probe_confidence)
    usable_df["confidence_delta"] = usable_df["debate_confidence"] - usable_df["control_confidence"]

    non_flipped = usable_df[~usable_df["flipped"]].copy()
    deltas = non_flipped["confidence_delta"].values
    if len(deltas) >= 2:
        tstat, tp = stats.ttest_rel(non_flipped["debate_confidence"], non_flipped["control_confidence"])
    else:
        tstat, tp = np.nan, np.nan

    print(f"\nNon-flipped cases: {len(non_flipped)}")
    print(f"Mean confidence delta (debate - control): {deltas.mean():+.3f}" if len(deltas) else "No non-flipped cases.")
    print(f"Paired t-test: t={tstat:.3f}, p={tp:.4f}")
    print(f"Share with a meaningful confidence drop (delta < -0.05): "
          f"{(non_flipped['confidence_delta'] < -0.05).mean():.1%}" if len(deltas) else "n/a")

    # Result 3: Layer-wise confidence trajectory, control vs debate
    debate_trajs = np.stack(usable_df["debate_hs"].apply(lambda hs: layerwise_trajectory(hs, layer_results)).values)
    control_trajs = np.stack(usable_df["control_hs"].apply(lambda hs: layerwise_trajectory(hs, layer_results)).values)
    mean_debate_traj = debate_trajs.mean(axis=0)
    mean_control_traj = control_trajs.mean(axis=0)

    # wrong-tutor confidence/strength vs whether it flipped the student
    wrong_tutor_strength_by_flip = usable_df.groupby("flipped")["wrong_tutor_confidence"].mean()
    print("\nMean wrong-tutor (Tutor A) self-confidence, by whether it flipped Tutor B:")
    print(wrong_tutor_strength_by_flip)

    corr_strength_flip = np.corrcoef(usable_df["wrong_tutor_confidence"], usable_df["flipped"].astype(int))[0, 1]
    print(f"Correlation between wrong-tutor confidence and flip: {corr_strength_flip:+.3f}")

    #Plots & Exports
    sns.set_theme(style="whitegrid")

    # Figure 2: flip rate by domain
    fig, ax = plt.subplots(figsize=(7, 5))
    flip_rate_by_domain.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#c0392b")
    ax.axhline(flip_rate_overall, color="black", linestyle="--", label=f"Overall ({flip_rate_overall:.1%})")
    ax.set_ylabel("Behavioral flip rate")
    ax.set_title(f"Flip rate by domain -- {MODEL_KEY} (n_usable={n_usable})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_flip_rate_by_domain.png"), dpi=200)
    plt.show()

    # Figure 3: confidence drop, control vs debate (non-flipped cases)
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(non_flipped) > 0:
        plot_df = non_flipped.melt(
            id_vars=["question"], value_vars=["control_confidence", "debate_confidence"],
            var_name="condition", value_name="probe_confidence"
        )
        sns.stripplot(data=plot_df, x="condition", y="probe_confidence", ax=ax, size=8, alpha=0.7)
        sns.pointplot(data=plot_df, x="condition", y="probe_confidence", ax=ax, color="black", errorbar="ci", markers="D")
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.5)
        ax.set_title(f"Silent capitulation: P(correct) in non-flipped cases\n"
                     f"mean delta={deltas.mean():+.3f}, paired t-test p={tp:.3f}")
    else:
        ax.text(0.5, 0.5, "No non-flipped usable cases", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_confidence_drop.png"), dpi=200)
    plt.show()

    # Figure 4: layer-wise confidence trajectory, control vs debate
    fig, ax = plt.subplots(figsize=(8, 5))
    layers_x = list(range(M.NUM_HIDDEN_STATES))
    ax.plot(layers_x, mean_control_traj, marker="o", label="Control", color="#2980b9")
    ax.plot(layers_x, mean_debate_traj, marker="o", label="Debate", color="#c0392b")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(best_layer, color="green", linestyle="--", alpha=0.5, label=f"Probe layer ({best_layer})")
    ax.set_xlabel("Hidden-state index (0=embeddings)"); ax.set_ylabel("Mean probe confidence P(correct)")
    ax.set_title(f"Layer-wise probe confidence: control vs debate -- {MODEL_KEY}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_layer_confidence.png"), dpi=200)
    plt.show()

    #Exports
    layer_acc_df.to_csv(os.path.join(TABLE_DIR, "probe_accuracy.csv"), index=False)
    debate_export = debate_df.drop(columns=["debate_hs", "control_hs"])
    debate_export.to_csv(os.path.join(TABLE_DIR, "debate_results.csv"), index=False)

    summary = {
        "model_key": MODEL_KEY,
        "model_name": MODEL_NAME,
        "num_layers": M.NUM_LAYERS,
        "best_probe_layer": best_layer,
        "probe_held_out_acc": float(real_acc),
        "probe_shuffled_acc": float(shuf_mean),
        "n_truth_probe_items": int(n_truth_items),
        "n_debate_items_total": n_total,
        "n_debate_items_usable": n_usable,
        "natural_disagreement_rate": float(debate_df["natural_disagreement"].mean()),
        "behavioral_flip_rate_overall": float(flip_rate_overall),
        "behavioral_flip_rate_by_domain": flip_rate_by_domain.to_dict(),
        "behavioral_flip_rate_by_difficulty": flip_rate_by_difficulty.to_dict(),
        "n_non_flipped": int(len(non_flipped)),
        "silent_capitulation_mean_delta": float(deltas.mean()) if len(deltas) else None,
        "silent_capitulation_ttest_p": float(tp) if not np.isnan(tp) else None,
        "wrong_tutor_confidence_by_flip": {str(k): float(v) for k, v in wrong_tutor_strength_by_flip.items()},
        "wrong_tutor_confidence_flip_correlation": float(corr_strength_flip),
    }
    with open(os.path.join(SUMMARY_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nAll figures, CSVs, and summary.json saved to: results/")
    print(f"\nTo compare a different model: set MODEL_KEY to one of "
          f"{list(MODEL_CANDIDATES.keys())} in config.py and re-run.")
    return summary
