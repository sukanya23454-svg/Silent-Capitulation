import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from silent_capitulation.model import load_model
from silent_capitulation.truth_dataset import build_truth_items, collect_probe_data
from silent_capitulation.probe_training import run_probe_validation, plot_probe_accuracy
from silent_capitulation.debate_dataset import build_debate_items
from silent_capitulation.debate_simulation import run_debates
from silent_capitulation.analysis import run_analysis

load_model()

all_truth_items = build_truth_items()
probe_X, probe_y, probe_domain, probe_difficulty = collect_probe_data(all_truth_items)

layer_results, layer_acc_df, best_layer, best_probe, real_acc, shuf_mean = run_probe_validation(probe_X, probe_y)
plot_probe_accuracy(layer_acc_df, shuf_mean, best_layer)

debate_items = build_debate_items()
debate_df = run_debates(debate_items, best_probe, best_layer)

run_analysis(debate_df, layer_results, layer_acc_df, best_layer, best_probe,
             real_acc, shuf_mean, len(all_truth_items))
