import tsplib95 as tsp
import numpy as np

filename = 'kroA100.tsp'

p: tsp.Problem = tsp.load_problem(filename)

no_of_nodes = len(p.node_coords)
adjacency_matrix = np.zeros((no_of_nodes, no_of_nodes)).astype(int)


def euclidean_distance(node1, node2):
    return int(((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5 + 0.5)


for node1 in range(no_of_nodes):
    for node2 in range(no_of_nodes):
        distance = euclidean_distance(p.node_coords[node1 + 1], p.node_coords[node2 + 1])
        adjacency_matrix[node1][node2] = distance
