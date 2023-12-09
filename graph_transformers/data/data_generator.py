import numpy as np
import networkx as nx
import random
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


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
        
        while len(self.graphs) < num_samples:
            adjacency_matrix, edge_weights = self.generate_graph(num_nodes, edge_prob, max_weight)
            if (adjacency_matrix.sum(axis = 1) == 0).any():
                continue
            graph_data = torch.stack([
                torch.from_numpy(adjacency_matrix), 
                torch.from_numpy(edge_weights)
                ], dim=0)  # Shape: (2, n, n)
            self.graphs.append(graph_data)

            if self.target_type == "shortest_path":
                self.targets.append(torch.tensor(self.shortest_path_length(edge_weights)))
            elif self.target_type == "min_coloring":
                self.targets.append(torch.tensor([self.min_coloring(adjacency_matrix, 1000)[0]]))


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]


    def generate_graph(self, n, p, max_weight=10, neg_weights = False):
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

    def shortest_path_length(self, edge_weights):
        """
        Compute the shortest path length between node 0 and 1 in the graph defined by the edge weights.

        Args:
            edge_weights (np.array): Edge weights

        Returns:
            path_length (int): Shortest path length
        """
        # edge_weights = np.where(adj_matrix == 1, edge_weights, float("+inf")) # Set non-edges to infinity
        G = nx.from_numpy_array(edge_weights)
        
        path_lengths = nx.single_source_dijkstra_path_length(G, source=0, weight='weight')

        # Create a numpy array of path lengths
        path_length_array = np.full(self.num_nodes, -1)
        for node, length in path_lengths.items():
            path_length_array[node] = length

        return path_length_array


    def is_valid_coloring(self, graph, colors):
        for node in range(len(graph)):
            for neighbor in range(len(graph)):
                if graph[node][neighbor] and colors[node] == colors[neighbor]:
                    return False
        return True

    def color_graph(self, graph, m, colors, node, iteration, max_iterations):
        if iteration > max_iterations:
            return False, iteration

        if node == len(graph):
            return self.is_valid_coloring(graph, colors), iteration

        for color in range(1, m + 1):
            colors[node] = color
            valid, iteration = self.color_graph(graph, m, colors, node + 1, iteration + 1, max_iterations)
            if valid:
                return True, iteration

        return False, iteration

    def min_coloring(self, adjacency_matrix, max_iterations):
        n = len(adjacency_matrix)
        for m in range(1, n + 1):
            colors = [0] * n
            found, iteration = self.color_graph(adjacency_matrix, m, colors, 0, 0, max_iterations)
            if found:
                return m, iteration
        return -1, iteration

    
    def plot_graph(self, idx: None | int = None, adjacency_matrix : None | np.ndarray = None , edge_weights_matrix : None | np.ndarray = None):
        """
        Plot a graph given its adjacency matrix.

        Args:
            idx (int): Index of the graph in the dataset
            adjacency_matrix (np.array): Adjacency matrix of the graph
        """
        if idx is not None:
            adjacency_matrix = self.graphs[idx][0].numpy()
            edge_weights_matrix = self.graphs[idx][1].numpy()
        
        elif adjacency_matrix is None:
            raise ValueError("Either idx or adjacency_matrix must be provided.")
        
        # Convert the adjacency matrix to a NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix)

        # Generate layout for visualization
        pos = nx.spring_layout(G)

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, edge_color='gray', linewidths=1, 
                font_size=15)

        # If edge weights matrix is provided, add edge labels
        if edge_weights_matrix is not None:
            edge_labels = {(i, j): edge_weights_matrix[i][j] for i, j in G.edges if edge_weights_matrix[i][j] != 0}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Display the plot
        plt.show()

if __name__ == "__main__":
    # Example usage
    n = 5  # Number of nodes
    p = 0.5  # Edge probability
    np.random.seed(1)
    random.seed(1)

    # adj_matrix, edge_weights = GraphDataset.generate_graph(n, p)
    # target = GraphDataset.shortest_path_length(edge_weights)

    # print("Adjacency Matrix:\n", adj_matrix)
    # print("Edge Weights:\n", edge_weights)
    # print("Shortest Path Length from Node 0 to 1:", target)

    data = GraphDataset(num_samples=10, num_nodes=n, edge_prob=p, target_type="shortest_path", max_weight=10, graph_neg_weights=False, seed=1)


    test = np.array([
        [0., 0., 0., 0., 1., 0., 1.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 0.],
        [1., 0., 0., 1., 0., 1., 0.],
        [0., 0., 1., 0., 1., 0., 1.],
        [1., 0., 0., 0., 0., 1., 0.]
    ])

    exit(1)