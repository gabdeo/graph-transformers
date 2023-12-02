import numpy as np
import networkx as nx
import random
import torch
from torch.utils.data import Dataset


class GraphDataset(Dataset):

    def __init__(self, num_samples, num_nodes, edge_prob, target_type = "shortest_path", max_weight=10, graph_neg_weights=False, seed=None):
        """
        Dataset class for generating random graphs.

        Args:
            num_samples (int): Number of samples
            num_nodes (int): Number of nodes
            edge_prob (float): Edge probability
            target_type (str): Target type (shortest path)
            max_weight (int): Maximum weight for the edges
            neg_weights (bool): If True, generate negative weights
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(1)
        
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.max_weight = max_weight
        self.neg_weights = graph_neg_weights
        self.target_type = target_type
        self.seed = seed

        self.graphs = []
        self.targets = []
        
        for _ in range(num_samples):
            adjacency_matrix, edge_weights = self.generate_graph(n, p, max_weight)
            graph_data = torch.stack([adjacency_matrix, edge_weights], dim=0)  # Shape: (2, n, n)
            self.graphs.append(graph_data)

            if self.target_type == "shortest_path":
                self.targets.append(self.shortest_path_length(edge_weights))


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


    def generate_graph(n, p, max_weight=10, neg_weights = False):
        """
        Generate a random graph with n nodes and edge probability p. Returns the adjacency matrix and edge weights.

        Args:
            n (int): Number of nodes
            p (float): Edge probability
            max_weight (int): Maximum weight for the edges
            neg_weights (bool): If True, generate negative weights
        
        Returns:
            adjacency (np.array): Adjacency matrix
            edge_weights (np.array): Edge weights
        """
        # Create a random graph + random weights
        graph = nx.erdos_renyi_graph(n, p)
        adjacency = nx.to_numpy_array(graph, dtype=np.int32)

        if neg_weights:
            edge_weights = np.random.randint(-max_weight, max_weight+1, size=(n, n))
        else:
            edge_weights = np.random.randint(1, max_weight+1, size=(n, n))
        
        edge_weights = np.tril(edge_weights) 
        edge_weights += edge_weights.T
        np.fill_diagonal(edge_weights, 0) 
        edge_weights = edge_weights * adjacency


        return adjacency, edge_weights

    def shortest_path_length(edge_weights):
        """
        Compute the shortest path length between node 0 and 1 in the graph defined by the edge weights.

        Args:
            edge_weights (np.array): Edge weights

        Returns:
            path_length (int): Shortest path length
        """
        # edge_weights = np.where(adj_matrix == 1, edge_weights, float("+inf")) # Set non-edges to infinity
        G = nx.from_numpy_array(edge_weights)
        try:
            # Compute shortest path using Dijkstra's algorithm
            path_length = nx.dijkstra_path_length(G, 0, 1)
        except nx.NetworkXNoPath:
            path_length = -1

        return path_length


if __name__ == "__main__":
    # Example usage
    n = 5  # Number of nodes
    p = 0.5  # Edge probability
    np.random.seed(1)
    random.seed(1)

    adj_matrix, edge_weights = GraphDataset.generate_graph(n, p)
    target = GraphDataset.shortest_path_length(edge_weights)

    # print("Adjacency Matrix:\n", adj_matrix)
    print("Edge Weights:\n", edge_weights)
    print("Shortest Path Length from Node 0 to 1:", target)

    test = np.array([
        [0., 0., 0., 0., 1., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 0.],
        [1., 0., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 1.],
        [1., 0., 0., 0., 0., 1., 0.]
    ])

