import torch
import tqdm
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from graph_transformers.models.base_transformer import Transformer
from graph_transformers.data import GraphDataset
from matplotlib import pyplot as plt
from abc import ABC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelTrainer(ABC):
    def __init__(self, dataset, batch_size, epochs, learning_rate, train_test_split, seed = None):
        if seed:
            torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self.split_dataset(train_test_split)

        # Initialize data loaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.epochs = epochs
        self.learning_rate = learning_rate

        # IMPLEMENT IN SUBCLASSES
        self.model = None
        self.model_type = None


    def split_dataset(self, train_test_split):
        train_size = int(train_test_split* len(self.dataset))
        val_size = (len(self.dataset) - train_size) // 2
        test_size = len(self.dataset) - train_size - val_size
        return torch.utils.data.random_split(self.dataset, [train_size, val_size, test_size])



    def train(self, plot = True):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # lr and optimizwer fixed for now

        train_loss_hist = []
        val_loss_hist = []
        val_acc_hist = []

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_loss = self.train_loop(self.train_dataloader, loss_fn, optimizer)
            val_loss, val_acc = self.evaluate(self.val_dataloader, loss_fn)

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            val_acc_hist.append(val_acc)

            print(f"Train loss: {train_loss:>7f}, Val loss: {val_loss:>7f}, Val acc: {val_acc:>7f}")
        
        if plot:
            self.plot_loss(train_loss_hist, val_loss_hist, val_acc_hist)

        return train_loss_hist, val_loss_hist, val_acc_hist


    def plot_loss(self, train_loss_hist, val_loss_hist, val_acc_hist, save_path = None):
        fig, ax1 = plt.subplots()

        # Plot for training and validation loss
        ax1.plot(train_loss_hist, label="Train Loss", linestyle='-', color='b')
        ax1.plot(val_loss_hist, label="Val Loss", linestyle='-', color='g')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc='upper left')

        # Create a second y-axis for validation accuracy
        ax2 = ax1.twinx()
        ax2.plot(val_acc_hist, label="Val Acc", linestyle=':', color='r')
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc='upper right')

        plt.title("Training Loss and Validation Accuracy")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def pass_data(self, X, y):
        X, y = X.to(device), y.to(device)
        return self.model(X)

    def train_loop(self, dataloader, loss_fn, optimizer):

        self.model.train()
        for _ , (X, y) in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):

            # Compute prediction and loss
            pred = self.pass_data(X, y)
            loss = loss_fn(pred, y.type(torch.float32))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def evaluate(self, dataloader = None, loss_fn = nn.MSELoss()):
        if dataloader is None:
            dataloader = self.test_dataloader
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.pass_data(X, y)
                loss += loss_fn(pred, y.type(torch.float32)).item()
                correct += (pred.round() == y).type(torch.float32).sum().item()
        loss /= num_batches
        correct /= size
        # print(f"Avg loss: {loss:>8f} \n")
        # print(f"Accuracy: {correct:>8f} \n")

        return loss, correct

    def plot_predictions(self, dataloader = None, num_samples = 5):
        if dataloader is None:
            dataloader = self.test_dataloader

        self.model.eval()
        i = 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.pass_data(X, y)
                for i in range(num_samples):
                    if i > num_samples:
                        break
                    i += 1
                    
                    print(f"Target: {y[i].item():>7f}, Prediction: {pred[i].item():>7f}")
                    # Plots the graph
                    self.dataset.plot_graph(idx=i)

                
                if i > num_samples:
                    break

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


class TransformerTrainer(ModelTrainer):
    def __init__(self, dataset, batch_size, epochs, learning_rate, train_test_split, dim, attn_dim, mlp_dim,
                 num_heads, num_layers, attn_mask = None, skip_connexion = "sum", seq_len=None, out_dim=None, seed = None):
        super().__init__(dataset, batch_size, epochs, learning_rate, train_test_split, seed)

        # Initialize the Transformer model with specified parameters
        self.model = Transformer(dim=dim, 
                                 attn_dim=attn_dim, 
                                 mlp_dim=mlp_dim, 
                                 num_heads=num_heads, 
                                 num_layers=num_layers, 
                                 seq_len=seq_len, 
                                 out_dim=out_dim,
                                 skip_connexion=skip_connexion).to(self.device)

        self.model_type = 'transformer'
        self.attn_mask = attn_mask if attn_mask else torch.ones(self.dataset.num_nodes, self.dataset.num_nodes)

    def pass_data(self, X, y):
        batch_size = X.shape[0]
        attn_mask_batched = self.attn_mask.repeat(batch_size, 1, 1)

        X = X.permute((0, 2, 1, 3)).flatten(2, 3).type(torch.float32)
        X, y = X.to(self.device), y.to(self.device)

        # Compute prediction and loss
        pred, _ = self.model(X, attn_mask_batched)
        return pred


class MLPTrainer(ModelTrainer):
    def __init__(self, dataset, batch_size, epochs, learning_rate, train_test_split, layers_dim, num_layers, out_dim = None, seed = None):
        super().__init__(dataset, batch_size, epochs, learning_rate, train_test_split, seed)

        # Initialize the MLP
        for layer in range(num_layers):
            if layer == 0:
                self.model = nn.Linear(dataset.num_nodes * dataset.channels, layers_dim[layer])
            else:
                self.model = nn.Linear(layers_dim[layer-1], layers_dim[layer])
            self.model = nn.Sequential(self.model, nn.ReLU())
        
        if out_dim:
            self.model = nn.Linear(layers_dim[-1], out_dim)
        else:
            self.model = nn.Linear(layers_dim[-1], layers_dim)

        self.model_type = 'mlp'

    def pass_data(self, X, y):
        X = X.flatten(1, 2, 3).type(torch.float32)
        X, y = X.to(self.device), y.to(self.device)

        # Compute prediction and loss
        pred = self.model(X)
        return pred

class GNNTrainer:
    pass

if __name__ == "__main__":
        # Dataset params
    n = 10  # Number of nodes in the graph
    seq_len = n # Token sequence length
    channels = 2 # Number of edge features
    edge_prob = 0.5
    target_type = "shortest_path"
    max_weight = 10
    graph_neg_weights = False
    seed = 1
    num_samples = 50000

    # Model params
    dimension = n * channels  # Model dimension
    attn_dim = 64  # Model dimension
    mlp_dim = 64 # Hidden layer dimension
    n_heads = 8  # Number of transformer heads
    num_layers = 6  # Number of transformer layers
    train_test_split = 0.6 
    seed = 42 
    batch_size = 128 
    epochs = 64 
    lr = 0.001 

    dataset = GraphDataset(num_samples=num_samples, num_nodes=n, edge_prob=edge_prob, target_type=target_type, 
                           max_weight=max_weight, graph_neg_weights=graph_neg_weights, seed=seed)
    trainer = TransformerTrainer(dataset, batch_size, epochs, lr, train_test_split, dimension, attn_dim, mlp_dim, 
                                 n_heads, num_layers, attn_mask = None, seq_len=seq_len, out_dim=1, seed = seed)
    trainer.train()
    test_loss, test_acc = trainer.evaluate()
    print(f"Test loss: {test_loss:>7f}, Test acc: {test_acc:>7f}")
    trainer.save_model("models/transformer.pt")

    exit(1)