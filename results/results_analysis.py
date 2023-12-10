from matplotlib import pyplot as plt
from matplotlib import cm
import pickle

def compare_metric(paths, metric, color_gradient, title = None, save_path = None):
    """
    Args:
        paths (list): List of paths to the results.pkl files
        metric (str): Metric to compare (train_loss, val_loss, val_acc, test_loss, test_acc)
        color_gradient (str): Color gradient to use for the lines (blue, green, orange, etc.)
    """
    colormap = cm.get_cmap(color_gradient, len(paths) + 1)
    dottype = ["-","--","-.",":", "-","--","-.",":"]

    plt.clf()
    # plt.figure(figsize=(10, 8))
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            results = pickle.load(f)
        plt.plot(results[metric], color=colormap(i), label=path.split("/")[-2], linestyle = dottype[i])
    
    if title:
        plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    

if __name__ == "__main__":
    gradient_map = {"train_loss": "winter", "val_loss": "summer", "val_acc": "autumn"}

    
    for metric in ["train_loss", "val_loss", "val_acc"]:
        # Compare MLP models
        title = f"{' '.join(metric.split('_')).capitalize()} for different sizes of MLPs"
        compare_metric(
            [
                "results/shortest_path/mlp_small/results.pkl",
                "results/shortest_path/mlp_mid/results.pkl",
                "results/shortest_path/mlp_large/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/mlp_{metric}.svg",
        )

        # Compare Transformer models
        title = f"{' '.join(metric.split('_')).capitalize()} for different sizes of Transformers"
        compare_metric(
            [
                "results/shortest_path/transformer_small/results.pkl",
                "results/shortest_path/transformer_mid/results.pkl",
                "results/shortest_path/transformer_large/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/transformer_{metric}.svg",
        )

        # Compare GNN models
        title = f"{' '.join(metric.split('_')).capitalize()} for different sizes of GNNs"
        compare_metric(
            [
                "results/shortest_path/gnn_small/results.pkl",
                "results/shortest_path/gnn_mid/results.pkl",
                "results/shortest_path/gnn_large/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/gnn_{metric}.svg",
        )

        # Compare medium Transformer with and without attention mask
        title = f"{' '.join(metric.split('_')).capitalize()} for (medium) Transformer with and without attention mask"
        compare_metric(
            [
                "results/shortest_path/transformer_mid/results.pkl",
                "results/shortest_path/transformer_attn_mask/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/transformer_{metric}_mask.svg",
        )

        # Compare medium Transformer with and without positional encoding
        title = f"{' '.join(metric.split('_')).capitalize()} for Transformer with and without positional encoding"
        compare_metric(
            [
                "results/shortest_path/transformer_mid/results.pkl",
                "results/shortest_path/transformer_pos_enc/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/transformer_{metric}_pos_enc.svg",
        )

        # Compare medium Transformer with and without skip connexion
        title = f"{' '.join(metric.split('_')).capitalize()} for Transformer with and without skip connexion"
        compare_metric(
            [
                "results/shortest_path/transformer_mid/results.pkl",
                "results/shortest_path/transformer_skip_connexion/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/transformer_{metric}_skip_connexion.svg",
        )

        # Compare all transformer models
        title = f"{' '.join(metric.split('_')).capitalize()} for all Transformer variations"
        compare_metric(
            [
                "results/shortest_path/transformer_small/results.pkl",
                "results/shortest_path/transformer_mid/results.pkl",
                "results/shortest_path/transformer_attn_mask/results.pkl",
                "results/shortest_path/transformer_pos_enc/results.pkl",
                "results/shortest_path/transformer_skip_connexion/results.pkl",
            ],
            metric,
            gradient_map[metric],
            title = title,
            save_path=f"results/comparative_plots/transformer_{metric}_all.svg",
        )