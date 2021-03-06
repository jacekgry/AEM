import itertools
import operator
import sys

import matplotlib.pyplot as plt
import numpy as np
import tsplib95 as tsp


def euclidean_distance(node1, node2):
    return int(((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5 + 0.5)


def make_adjacency_matrix(nodes):
    no_of_nodes = len(nodes)
    adjacency_matrix = np.zeros((no_of_nodes, no_of_nodes)).astype(int)
    for node1 in range(no_of_nodes):
        for node2 in range(no_of_nodes):
            distance = euclidean_distance(nodes[node1 + 1], nodes[node2 + 1])
            adjacency_matrix[node1][node2] = distance
    return adjacency_matrix


def get_path_length(matrix, order):
    length = 0
    for i in range(len(order) - 1):
        length += matrix[order[i]][order[i + 1]]
    return length + matrix[order[-1]][order[0]]


def find_combinations(matrix, order, node):
    comb = []
    for z in range(1, len(order) + 1):
        combination = order[:]
        combination.insert(z, node)
        value = get_path_length(matrix, combination)
        comb.append((combination, value))
    return sorted(comb, key=operator.itemgetter(1))


def greedy_cycle_best_comb(matrix, order, node):
    comb = find_combinations(matrix, order, node)
    return min(comb, key=operator.itemgetter(1))


def save_graph(nodes, order, graph_filename):
    plt.clf()

    x = []
    y = []
    for i in range(1, len(nodes) + 1):
        x.append(nodes.get(i)[0])
        y.append(nodes.get(i)[1])

    new_x = []
    new_y = []
    for i in order:
        new_x.append(x[i])
        new_y.append(y[i])
    new_x.append(x[order[0]])
    new_y.append(y[order[0]])

    plt.scatter(x, y, color='blue')
    plt.plot(new_x, new_y, color='red')
    # plt.show()
    plt.savefig(f'{instance_name}/{graph_filename}.png')


def make_regret_comb(k, matrix, order, node):
    comb = find_combinations(matrix, order, node)
    regret = 0
    for i in comb[:k + 1]:
        regret += i[1] - comb[0][1]
    return [comb, regret]


def greedy_cycle(start_node):
    order_list = [start_node]

    for i in range(cycle_size - 1):
        temp = []
        for t in range(len(problem.node_coords)):
            if t not in order_list:
                temp.append(greedy_cycle_best_comb(adjacency_matrix, order_list, t))
        order_list, _ = min(temp, key=operator.itemgetter(1))
    save_graph(problem.node_coords, order_list, f'greedy{start_node}')
    return order_list


def regret(start_node, k=1):
    order_list = [start_node]

    for i in range(cycle_size - 1):
        temp = []
        for t in range(len(problem.node_coords)):
            if t not in order_list:
                temp.append(make_regret_comb(k, adjacency_matrix, order_list, t))

        order_list = max(temp, key=operator.itemgetter(1))[0][0][0]
    save_graph(problem.node_coords, order_list, f'regret{start_node}')
    return order_list


def delta_of_swap(vertex1, vertex2, neighbors1, neighbors2=None):
    if neighbors2 is None:
        neighbors2 = []

    new_dist = 0
    old_dist = 0

    for p in neighbors1:
        if p not in (vertex2, vertex1):
            new_dist += adjacency_matrix[vertex2][p]
            old_dist += adjacency_matrix[vertex1][p]

    for p in neighbors2:
        if p not in (vertex2, vertex1):
            new_dist += adjacency_matrix[vertex1][p]
            old_dist += adjacency_matrix[vertex2][p]

    return - old_dist + new_dist


def delta_edges(neighbors, start, end):
    new_dist = adjacency_matrix[start][neighbors[1]] + adjacency_matrix[end][neighbors[0]]
    old_dist = adjacency_matrix[start][neighbors[0]] + adjacency_matrix[end][neighbors[1]]
    return new_dist - old_dist


# wewnatrz - wierzcholki
def greedy_local_search_vertices(start_sol, unused_vertices, start_dist):
    n = len(start_sol)
    no_of_neighs = int(n ** 2 + n * (n - 1) / 2)
    external_neighs_indexes = np.random.choice(no_of_neighs, n ** 2, replace=False)
    choices = np.zeros(no_of_neighs)
    choices[external_neighs_indexes] = 1

    best_sol, best_distance = start_sol.copy(), start_dist
    current_unused_vertices = unused_vertices.copy()
    current_sol = start_sol.copy()
    current_dist = start_dist
    improved = True

    while improved:
        int_neighs = iter([(i, j) for i in range(cycle_size) for j in range(i + 1, cycle_size)])
        ext_neighs = itertools.product(range(cycle_size), enumerate(current_unused_vertices))
        improved = False
        for neigh_no in range(no_of_neighs):
            if choices[neigh_no] == 0:
                index, idx_of_int = int_neighs.__next__()
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                neighbors2 = [current_sol[(idx_of_int - 1) % cycle_size],
                              current_sol[(idx_of_int + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], current_sol[idx_of_int], neighbors1, neighbors2)
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    best_sol[index], best_sol[idx_of_int] = current_sol[idx_of_int], current_sol[index]
                    best_distance = current_dist + delta
                    improved = True
                    break
            else:
                index, (ext_idx, ext_vertex) = ext_neighs.__next__()
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    best_sol[index] = ext_vertex
                    best_distance = current_dist + delta
                    improved = True
                    break
        current_sol, current_dist = best_sol.copy(), best_distance
        current_unused_vertices = [x for x in range(100) if x not in current_sol]
    return current_sol, current_dist


def steepest_local_search_vertices(start_sol, unused_vertices, start_dist):
    best_sol, best_distance = start_sol.copy(), start_dist
    current_unused_vertices = unused_vertices.copy()
    improved = True
    current_sol = start_sol.copy()
    current_dist = start_dist
    while improved:
        improved = False
        for index in range(cycle_size):
            for idx_of_int in range(index + 1, cycle_size):
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                neighbors2 = [current_sol[(idx_of_int - 1) % cycle_size],
                              current_sol[(idx_of_int + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], current_sol[idx_of_int], neighbors1, neighbors2)
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    best_sol[index], best_sol[idx_of_int] = current_sol[idx_of_int], current_sol[index]
                    best_distance = current_dist + delta
                    improved = True
            for ext_idx, ext_vertex in enumerate(current_unused_vertices):
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    best_sol[index] = ext_vertex
                    best_distance = current_dist + delta
                    improved = True

        current_sol, current_dist = best_sol.copy(), best_distance
        current_unused_vertices = [x for x in range(100) if x not in current_sol]
    return current_sol, current_dist


def greedy_local_search_edges(start_sol, unused_vertices, start_dist):
    n = len(start_sol)
    no_of_neighs = int(n ** 2 + n * (n - 3) / 2)
    external_neighs_indexes = np.random.choice(no_of_neighs, n ** 2, replace=False)
    choices = np.zeros(no_of_neighs)
    choices[external_neighs_indexes] = 1

    best_sol, best_distance = start_sol.copy(), start_dist
    current_unused_vertices = unused_vertices.copy()
    current_sol = start_sol.copy()
    current_dist = start_dist
    improved = True

    while improved:
        int_neighs = iter([(i, j) for i in range(cycle_size) for j in range(i + 1, cycle_size - 1)])
        ext_neighs = itertools.product(range(cycle_size), enumerate(current_unused_vertices))
        improved = False
        for neigh_no in range(no_of_neighs):
            if choices[neigh_no] == 0:
                index, idx_of_int = int_neighs.__next__()
                neighbors = [current_sol[(index - 1) % cycle_size], current_sol[(idx_of_int + 1) % cycle_size]]
                delta = delta_edges(neighbors, current_sol[index], current_sol[idx_of_int])
                if current_dist + delta < best_distance:
                    if (idx_of_int + 1) % cycle_size == 0:
                        best_sol[index::] = best_sol[index::][::-1]
                    else:
                        best_sol[index:(idx_of_int + 1) % cycle_size] = best_sol[index:(idx_of_int + 1) % cycle_size][
                                                                        ::-1]
                    best_distance = current_dist + delta
                    improved = True
                    break

            else:
                index, (ext_idx, ext_vertex) = ext_neighs.__next__()
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    best_sol[index] = ext_vertex
                    best_distance = current_dist + delta
                    improved = True
                    break
        current_sol, current_dist = best_sol.copy(), best_distance
        current_unused_vertices = [x for x in range(100) if x not in current_sol]
    return current_sol, current_dist


def steepest_local_search_edges(start_sol, unused_vertices, start_dist):
    best_sol, best_distance = start_sol.copy(), start_dist
    current_unused_vertices = unused_vertices.copy()
    improved = True
    current_sol = start_sol.copy()
    current_dist = start_dist
    while improved:
        improved = False
        for index in range(cycle_size):
            for i in range(index + 1, cycle_size - 1):
                neighbors = [current_sol[(index - 1) % cycle_size], current_sol[(i + 1) % cycle_size]]
                delta = delta_edges(neighbors, current_sol[index], current_sol[i])
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    if (i + 1) % cycle_size == 0:
                        best_sol[index::] = best_sol[index::][::-1]
                    else:
                        best_sol[index:i + 1] = best_sol[index:i + 1][::-1]
                    best_distance = current_dist + delta
                    improved = True

            for ext_idx, ext_vertex in enumerate(current_unused_vertices):
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
                if current_dist + delta < best_distance:
                    best_sol = current_sol.copy()
                    best_sol[index] = ext_vertex
                    best_distance = current_dist + delta
                    improved = True
        current_sol, current_dist = best_sol.copy(), best_distance
        current_unused_vertices = [x for x in range(100) if x not in current_sol]
    return current_sol, current_dist


instance_name = sys.argv[1] if len(sys.argv) == 2 else 'kroA100'
problem: tsp.Problem = tsp.load_problem(f'{instance_name}.tsp')
cycle_size = round(int(np.ceil(len(problem.node_coords) / 2)))
adjacency_matrix = make_adjacency_matrix(problem.node_coords)
