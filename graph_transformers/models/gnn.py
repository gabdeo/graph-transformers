import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Batch
from graph_transformers.utils.graph_problems import dijkstra
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, depth, breadth):
        super(MLP, self).__init__()
        if depth == 1:
            self.mlp = nn.Linear(input_dim, output_dim)
            return

        layers = [nn.Linear(input_dim, breadth), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(breadth, breadth))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(breadth, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class SimpleGNNLayer(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, mlp_hidden_layer_size, mlp_n_layers, aggregation='mean'):
        super(SimpleGNNLayer, self).__init__()
        self.aggregation = aggregation

        # MLP to update node features based on neighboring node and edge features
        self.node_edge_mlp = MLP(node_feature_size + node_feature_size + edge_feature_size, node_feature_size, mlp_n_layers, mlp_hidden_layer_size)

        # MLP to map aggregated feature to new node feature
        self.aggregate_mlp = MLP(node_feature_size, node_feature_size, mlp_n_layers, mlp_hidden_layer_size)

    def forward(self, features, adj_matrix):
        # node_features: Tensor of shape (batch, n, n, node_pair_feature)
        # edge_features: Tensor of shape (batch, n, n, edge_feature_size)
        # adj_matrix: Tensor of shape (batch, n, n), binary adjacency matrix

        batch_size, num_nodes, _, _= features.size()


        mlp_input = features.flatten(0,-2)
        node_output = self.node_edge_mlp(mlp_input)
        node_output = node_output.view(batch_size, num_nodes, num_nodes, -1)[...,0]

        # Aggregate features
        if self.aggregation == 'mean':
            node_output_masked = node_output * adj_matrix
            aggregated_features = torch.sum(node_output_masked, dim=-1) / torch.sum(adj_matrix, dim=-1)
        elif self.aggregation == 'min':
            node_output_masked = torch.where(adj_matrix == 1, node_output, torch.ones_like(node_output) * float('inf')) 
            aggregated_features = torch.min(node_output_masked, dim=-1)[0]
        elif self.aggregation == 'max':
            node_output_masked = torch.where(adj_matrix == 1, node_output, torch.ones_like(node_output) * float('-inf')) 
            aggregated_features = torch.max(node_output_masked, dim=-1)[0]
        else:
            raise ValueError("Invalid aggregation method. Choose from 'mean', 'min', or 'max'.")
        
        # # Apply the linear layer
        # mlp_input_2 = aggregated_features.unsqueeze(-1).flatten(0,-2)

        aggregated_features_1 = aggregated_features.unsqueeze(-1).repeat(1,1,num_nodes).unsqueeze(-1)
        updated_features = torch.cat([aggregated_features_1, aggregated_features_1.transpose(1,2), features[...,[2]]], dim=-1)

        return updated_features

class SimpleGNN(nn.Module):

    def __init__(
            self,
            n_iterations,
            node_feature_size, 
            edge_feature_size, 
            mlp_hidden_layer_size, 
            mlp_n_layers, 
            aggregation='mean'
        ):
        super(SimpleGNN, self).__init__()
        self.n_iterations = n_iterations
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.mlp_hidden_layer_size = mlp_hidden_layer_size
        self.mlp_n_layers = mlp_n_layers
        self.aggregation = aggregation

        self.gnn_layer = nn.ModuleList(
            [SimpleGNNLayer(node_feature_size, edge_feature_size, mlp_hidden_layer_size, mlp_n_layers, aggregation) for _ in range(n_iterations)]
        )        
    def forward(self, features, adj_matrix):
        # node_features: Tensor of shape (batch, n, node_feature_size)
        # edge_features: Tensor of shape (batch, n, n, edge_feature_size)
        # adj_matrix: Tensor of shape (batch, n, n), binary adjacency matrix

        for layer in self.gnn_layer:
            features = layer(features, adj_matrix)

        return features



class GNNLayer(MessagePassing):
    def __init__(
        self,
        use_x_i,
        aggr,
        message_mlp,
        update_mlp,
    ):
        super(GNNLayer, self).__init__(aggr=aggr)
        self.use_x_i = use_x_i
        self.message_mlp = message_mlp
        self.update_mlp = update_mlp

    def forward(self, batch):
        # Pass message to propagate
        return self.propagate(
            edge_index=batch.edge_index, x=batch.x, edge_weight=batch.edge_attr
        )

    def message(self, x_i, x_j, edge_weight):
        # Use x_i if use_x_i is True, otherwise just use x_j and edge_weight
        if self.use_x_i:
            message_input = torch.cat([x_i, x_j, edge_weight.unsqueeze(-1)], dim=-1)
        else:
            message_input = torch.cat([x_j, edge_weight.unsqueeze(-1)], dim=-1)
        return self.message_mlp(message_input)

    def update(self, aggr_out):
        # Update node features to the aggregated messages
        return self.update_mlp(aggr_out)


class GNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_iter,
        message_depth,
        message_breadth,
        update_depth,
        update_breadth,
        use_x_i=False,
        aggr="min",
        out_dim=None,
    ):
        super(GNN, self).__init__()
        self.num_iter = num_iter

        # Create common MLPs for message and update functions
        input_dim = 2 + use_x_i
        self.message_mlp = MLP(input_dim, 1, message_depth, message_breadth)
        self.update_mlp = MLP(1, 1, update_depth, update_breadth)

        # Create layers with references to the common MLPs
        self.layers = nn.ModuleList(
            [
                GNNLayer(use_x_i, aggr, self.message_mlp, self.update_mlp)
                for _ in range(num_iter)
            ]
        )
        if out_dim:
            self.output_mlp = MLP(num_nodes, out_dim, 2, 2 * num_nodes)
        self.out_dim = out_dim

    def forward(self, batch):
        # Ensure that batch is a Batch object
        if not isinstance(batch, Batch):
            raise TypeError("Input must be a PyTorch Geometric Batch object")

        for layer in self.layers:
            batch.x = layer(batch)

        # Each graph in the batch has the same number of nodes
        num_nodes_per_graph = batch.num_nodes // batch.num_graphs
        final_output = batch.x.view(batch.num_graphs, num_nodes_per_graph, -1).squeeze()

        if self.out_dim:
            final_output = self.output_mlp(final_output)

        return final_output


def matrix_to_graph(matrices):
    """
    Convert a batch of matrices into a DataLoader that yields Batch objects.

    Each pair of matrix in the batch represents a graph, where the pair of matrix contains
    both adjacency information and edge weights.

    Parameters:
    matrices (torch.Tensor): A batch of matrices with shape (B, c, n, n),
                             where B is the batch size, c is the number of channels
                             (c=2 for adjacency matrix and edge weights matrix),
                             and n is the number of nodes in each graph.
    batch_size (int): The size of each batch in the DataLoader.

    Returns:
    DataLoader: A DataLoader that yields Batch objects, each representing a batch of graphs.
    """
    graph_list = []
    for matrix in matrices[:, 1]:  # take only the edges weights
        num_nodes = matrix.size(0)
        edge_index = []
        edge_weight = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if matrix[i, j] != 0:  # Non-zero entries indicate edges
                    edge_index.append([i, j])
                    edge_weight.append(matrix[i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # Node features: Initialized as 1 for all nodes except the source node
        x = torch.full((num_nodes, 1), 1)
        x[0][0] = 0  # First node is the source

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        graph_list.append(graph_data)

    return Batch.from_data_list(graph_list)


def create_synthetic_data(batch_size, num_nodes):
    # Initialize the tensor for the batch of matrices
    edge_weight_matrices = torch.zeros((batch_size, num_nodes, num_nodes))

    for b in range(batch_size):
        # Create a random matrix for the upper triangle
        upper_triangle = torch.rand((num_nodes, num_nodes))

        # Ensure weights are either 0 or 1 for simplicity
        upper_triangle[upper_triangle < 0.5] = 0
        upper_triangle[upper_triangle >= 0.5] = 1

        # Make the matrix symmetric to represent an undirected graph
        symmetric_matrix = upper_triangle + upper_triangle.T
        symmetric_matrix[
            symmetric_matrix > 1
        ] = 1  # If any value is above 1, set it to 1

        # Ensure the diagonal is zero (no self-loops)
        symmetric_matrix.fill_diagonal_(0)

        edge_weight_matrices[b] = symmetric_matrix

    # Duplicate to simulate (B, 2, n, n) input with adjacency matrix and weights
    matrices = torch.stack([edge_weight_matrices, edge_weight_matrices], dim=1)

    return matrices


if __name__ == "__main__":
    # Creating synthetic data
    batch_size = 2
    num_nodes = 4
    synthetic_data = create_synthetic_data(batch_size, num_nodes)

    # Convert matrices to DataLoader of graph batches
    graph_loader = matrix_to_graph(synthetic_data, batch_size=batch_size)

    # Initialize the BFGNN model
    SETTINGS = {
        "num_nodes": 4,
        "num_iter": 4,  # For a graph with 4 nodes
        "message_depth": 3,
        "message_breadth": 16,
        "update_depth": 2,
        "update_breadth": 8,
        "use_x_i": True,
        "aggr": "min",
    }
    model = GNN(**SETTINGS)

    # Dummy target for loss calculation (in a real scenario, this should be meaningful)
    dummy_target = torch.rand((batch_size, num_nodes))

    # Example Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()  # Mean Squared Error Loss for demonstration

    for epoch in range(5):  # 5 epochs
        for batch in graph_loader:
            print(batch)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, dummy_target)  # Dummy loss calculation
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
