import numpy as np
from graph_transformers.utils.traintools import TrainResult, compute_accuracy
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from graph_transformers.utils.graph_problems import dijkstra


# Define a GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_nodes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_nodes, 16)
        self.conv2 = GCNConv(16, num_nodes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def adj_to_gnn_data(adj_matrix, use_edge_weights=True):
    """
    Convert an adjacency matrix to a Data object for GNN.

    Parameters:
    adj_matrix (np.array): 2D NumPy array representing the adjacency matrix.
    use_edge_weights (bool): Whether to use edge weights from the adjacency matrix.

    Returns:
    torch_geometric.data.Data: Data object for GNN.
    """
    edge_index = []
    edge_weight = []

    num_nodes = adj_matrix.shape[0]

    # Convert adjacency matrix to edge index and edge weights
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0:  # Assuming 0 means no edge
                edge_index.append([i, j])
                edge_weight.append(adj_matrix[i][j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if use_edge_weights:
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    else:
        edge_weight = None

    # Node features - one-hot encoding
    node_features = torch.eye(num_nodes)

    # Create the data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)

    return data


def train_gnn(
    model, graph_dataset, targets, num_epochs=200, learning_rate=0.01, print_interval=10
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_result = TrainResult(
        num_epochs, learning_rate, model, train_losses=[], train_accs=[]
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        for data in graph_dataset:  # Iterate over each graph in the dataset
            optimizer.zero_grad()

            output = model(data)
            loss = loss_fn(
                output, targets[data.graph_id]
            )  # Assuming targets are keyed by graph_id
            epoch_loss += loss.item()

            # Compute and store accuracies
            train_acc = compute_accuracy(output, targets[data.graph_id])
            epoch_acc += train_acc

            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(graph_dataset)
        avg_epoch_acc = epoch_acc / len(graph_dataset)
        train_result.train_losses.append(avg_epoch_loss)
        train_result.train_accs.append(avg_epoch_acc)

        if epoch % print_interval == 0:
            print(f"Epoch {epoch} | Loss: {avg_epoch_loss}")

    return train_result


if __name__ == "__main__":
    adj_matrix = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]])
    num_nodes = len(adj_matrix)
    model = GNN(num_nodes)
    data = adj_to_gnn_data(adj_matrix, use_edge_weights=True)

    # Compute target distances using Dijkstra's algorithm
    target_distances = dijkstra(adj_matrix)
    targets = torch.tensor(target_distances, dtype=torch.float)

    train_gnn(model, data, targets)
