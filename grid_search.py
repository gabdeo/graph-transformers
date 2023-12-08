import os
import torch
import tqdm
from torch import nn
import pickle
import numpy as np
from torch.utils.data import DataLoader
from graph_transformers import  GraphDataset, TransformerTrainer, MLPTrainer, GNNTrainer
from matplotlib import pyplot as plt
import json 


def create_dataset(dataset_config, task_type):
    dataset = GraphDataset(
        num_samples=dataset_config['num_samples'],
        num_nodes=dataset_config['n'],
        edge_prob=dataset_config['edge_prob'],
        target_type=task_type,
        max_weight=dataset_config['max_weight'],
        graph_neg_weights=dataset_config['graph_neg_weights'],
        seed=dataset_config['seed']
    )
    #Save dataset
    with open(f"datasets/dataset_{task_type}.pkl", 'wb') as f:
        pickle.dump(dataset, f)
    return dataset

def load_dataset(task_type):
    with open(f"datasets/dataset_{task_type}.pkl", 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def run_and_evaluate_model(model_trainer, plot = False, save_dir = None):
    train_loss_hist, val_loss_hist, val_acc_hist = model_trainer.train(plot = plot)
    test_loss, test_acc = model_trainer.evaluate()

    if save_dir:
        torch.save(model_trainer.model.state_dict(), os.path.join(save_dir, "model.pt"))
        # Save results
        with open(os.path.join(save_dir, "results.pkl"), 'wb') as f:
            pickle.dump({
                "train_loss": train_loss_hist,
                "val_loss": val_loss_hist,
                "val_acc": val_acc_hist,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, f)
        
        # Save loss plots:
        model_trainer.plot_loss(train_loss_hist, val_loss_hist, val_acc_hist, save_path = os.path.join(save_dir, "loss_plot.png"))

if __name__ == "__main__":
    
    tasks = ['shortest_path', 
             'min_coloring'
    ]
    sizes = ["small", 
             "mid", 
             "large"
    ]

    use_attn_mask = [False]


    for task in tasks:
        
        all_configs = json.load(open('configs.json', 'r')) 
        dataset_config = all_configs['dataset']

        if not os.path.exists(f"datasets/dataset_{task}.pkl"):
            dataset = create_dataset(dataset_config, task)
        else:
            dataset = load_dataset(task)
        
        models_config = all_configs['models']

        trans_config = all_configs['transformer'] | models_config
        trans_config["dim"] = dataset_config["n"] * dataset_config["channels"]
        
        if task == "min_coloring":
            trans_config["out_dim"] = 1 # Output is just chromatic number
        
        elif task == "shortest_path":
            trans_config ["out_dim"] = dataset_config["n"] # Output is shortest path at each node from origin node

        gnn_config = all_configs['gnn'] | models_config
        mlp_config = all_configs['mlp'] | models_config

        for size in sizes:
            if size == "large":
                mlp = MLPTrainer(dataset, **mlp_config)
                run_and_evaluate_model(mlp, plot = False, save_dir = f"results/{task}/mlp_large/")

            # gnn = GNNTrainer(dataset, **gnn_config)
            # run_and_evaluate_model(gnn, plot = False, save_dir = f"results/{task}/gnn_{size}/")

            transformer = TransformerTrainer(dataset, **trans_config, seq_len = dataset_config["n"])
            run_and_evaluate_model(transformer, plot = False, save_dir = f"results/{task}/transformer_{size}/")
        
        