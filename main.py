import tsplib95 as tsp
import numpy as np
import random
import operator
import matplotlib.pyplot as plt

filename = 'kroB100.tsp'

p: tsp.Problem = tsp.load_problem(filename)


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


def show_graph(nodes, order):
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

    plt.show()


def make_regret_comb(k, matrix, order, node):
    comb = find_combinations(matrix, order, node)
    regret = 0
    for i in comb[:k + 1]:
        regret += i[1] - comb[0][1]
    return [comb, regret]


def select_second_element():
    order_list = []
    start_node = random.choice(range(100))
    order_list.append(start_node)

    min_value = sorted(adjacency_matrix[order_list[0]])[1]
    index2 = adjacency_matrix[order_list[0]].tolist().index(min_value)

    order_list.append(index2)
    return order_list


def greedy_cycle():
    order_list = select_second_element()

    for i in range(cycle_size - 2):
        temp = []
        for t in range(len(p.node_coords)):
            if t not in order_list:
                temp.append(greedy_cycle_best_comb(adjacency_matrix, order_list, t))
        order_list, min_legth = min(temp, key=operator.itemgetter(1))
    show_graph(p.node_coords, order_list)
    return order_list


def regret(k):
    order_list = select_second_element()

    for i in range(cycle_size - 2):
        temp = []
        for t in range(len(p.node_coords)):
            if t not in order_list:
                temp.append(make_regret_comb(k, adjacency_matrix, order_list, t))

        order_list = max(temp, key=operator.itemgetter(1))[0][0][0]
    show_graph(p.node_coords, order_list)
    return order_list


cycle_size = round(int(np.ceil(len(p.node_coords) / 2)))
print('Cycle size: ', cycle_size)

adjacency_matrix = make_adjacency_matrix(p.node_coords)

# result = greedy_cycle()
result = regret(1)

print(f'Number of nodes in cycle: {len(result)}')
print(f'Cycle length: {get_path_length(adjacency_matrix, result)}')
