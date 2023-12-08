import numpy as np


def dijkstra(adj_matrix):
    """
    Compute the shortest path distances for all pairs of nodes in a graph.

    Parameters:
    adj_matrix (numpy.ndarray): A 2D numpy array representing the adjacency matrix of the graph.
                                The value at adj_matrix[i][j] represents the weight of the edge
                                from node i to node j, or infinity if there is no direct edge.

    Returns:
    numpy.ndarray: A 2D numpy array where the element at index [i][j] represents the shortest
                   path distance from node i to node j.
    """
    num_nodes = len(adj_matrix)
    distances = np.full((num_nodes, num_nodes), np.inf)
    for i in range(num_nodes):
        distances[i] = single_dijkstra(adj_matrix, i)
    return distances


def single_dijkstra(adj_matrix, start):
    """
    Compute the shortest path distances from a single start node to all other nodes in the graph.

    Parameters:
    adj_matrix (numpy.ndarray): A 2D numpy array representing the adjacency matrix of the graph.
    start (int): The index of the start node.

    Returns:
    numpy.ndarray: A 1D numpy array where the element at index [i] represents the shortest
                   path distance from the start node to node i.
    """
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    distances = np.full(num_nodes, np.inf)
    distances[start] = 0

    for _ in range(num_nodes):
        min_node = None
        for i in range(num_nodes):
            if not visited[i] and (
                min_node is None or distances[i] < distances[min_node]
            ):
                min_node = i

        if min_node is None:
            break

        visited[min_node] = True
        for i in range(num_nodes):
            if adj_matrix[min_node][i] and not visited[i]:
                new_dist = distances[min_node] + adj_matrix[min_node][i]
                if new_dist < distances[i]:
                    distances[i] = new_dist

    return distances
