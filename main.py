import tsplib95 as tsp
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import sys

instance_name = sys.argv[1] if len(sys.argv) == 2 else 'kroA100'
problem: tsp.Problem = tsp.load_problem(f'{instance_name}.tsp')
random.seed = 42


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
    plt.show()
    # plt.savefig(f'{instance_name}/{graph_filename}.png')


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


def delta_swap(new_point, distance, old_point, neighbors, neighbors2=None):
    new_dist = 0
    old_dist = 0
    if neighbors2 is not None:
        new_dist += adjacency_matrix[old_point][neighbors2[0]] + adjacency_matrix[old_point][neighbors2[1]]
        old_dist += adjacency_matrix[new_point][neighbors2[1]] + adjacency_matrix[new_point][neighbors2[0]]
    new_dist += adjacency_matrix[new_point][neighbors[1]] + adjacency_matrix[new_point][neighbors[0]]
    old_dist += adjacency_matrix[old_point][neighbors[0]] + adjacency_matrix[old_point][neighbors[1]]
    if new_dist < old_dist:
        return distance - old_dist + new_dist
    return distance


def find_firsold_pointt_better_point(rest_points, neighbors, point, index, distance):
    for index2, z in enumerate(rest_points):
        new_dist = delta_swap(z, distance, point, neighbors)
        if new_dist != distance:
            start_solution[index] = z
            rest_points[index2] = point
            return rest_points, start_solution, new_dist
    return rest_points, start_solution, distance


def find_best_point(rest_points, neighbors, index, distance):
    for index2 in range(len(rest_points)):
        new_dist = delta_swap(rest_points[index2], distance, start_solution[index], neighbors)
        if new_dist != distance:
            temp = start_solution[index]
            start_solution[index] = rest_points[index2]
            rest_points[index2] = temp
            distance = new_dist
    return rest_points, start_solution, distance


def find_best_swap(solution, index, distance, neighbors):
    for index2 in range(cycle_size):
        if index2 != index:
            neighbors2 = [solution[(index2-1) % cycle_size], solution[(index2+1) % cycle_size]]
            new_dist = delta_swap(start_solution[index2], distance, start_solution[index], neighbors, neighbors2)
            if new_dist != distance:
                temp = start_solution[index]
                start_solution[index] = start_solution[index2]
                start_solution[index2] = temp
                distance = new_dist
                if abs(index - index2) == 1:
                    neighbors = [solution[(index - 1) % cycle_size], solution[(index + 1) % cycle_size]]
    return start_solution, distance


def find_first_better_swap(solution, index, distance, neighbors):
    for index2 in range(len(start_solution)):
        if index2 != index:
            neighbors2 = [solution[(index2-1) % cycle_size], solution[(index2+1) % cycle_size]]
            new_dist = delta_swap(start_solution[index2], distance, start_solution[index], neighbors, neighbors2)
            if new_dist != distance:
                temp = start_solution[index]
                start_solution[index] = start_solution[index2]
                start_solution[index2] = temp
                distance = new_dist
                return start_solution, distance
    return start_solution, distance


def delta_edges(neighbors, start, end, distance):
    new_dist = adjacency_matrix[start][neighbors[1]] + adjacency_matrix[end][neighbors[0]]
    old_dist = adjacency_matrix[start][neighbors[0]] + adjacency_matrix[end][neighbors[1]]
    if new_dist < old_dist:
        return distance - old_dist + new_dist
    return distance


def find_first_better_edges(solution, index, distance):
    for i in range(index+1, cycle_size):
        neighbors = [solution[(index-1) % cycle_size], solution[(i+1) % cycle_size]]
        new_dist = delta_edges(neighbors, solution[index], solution[i], distance)
        if new_dist != distance:
            distance = new_dist
            solution[index:(i+1) % cycle_size] = solution[index:(i+1) % cycle_size][::-1]
            return solution, distance
    return solution, distance


def find_best_edges(solution, index, distance):
    for i in range(index+1, cycle_size):
        neighbors = [solution[(index-1) % cycle_size], solution[(i+1) % cycle_size]]
        new_dist = delta_edges(neighbors, solution[index], solution[i], distance)
        if new_dist != distance:
            distance = new_dist
            solution[index:(i+1) % cycle_size] = solution[index:(i+1) % cycle_size][::-1]
    return solution, distance


#wewnatrz - wierzcholki
def greedy_local_search_1(start_solution, rest_points, distance):
    choice = np.random.choice(2, cycle_size, replace=True)
    for index in range(cycle_size):
        neighbors = [start_solution[(index-1) % cycle_size], start_solution[(index+1) % cycle_size]]
        if choice[index] == 0:
            rest_points, start_solution, distance = find_first_better_point(rest_points, neighbors, start_solution[index], index, distance)
        else:
            start_solution, distance = find_first_better_swap(start_solution, index, distance, neighbors)
    print(distance)


#wewnatrz - wierzcholkowy
def steepest_local_search_1(start_solution, rest_points, distance):
    for index in range(cycle_size):
        neighbors = [start_solution[(index - 1) % cycle_size], start_solution[(index + 1) % cycle_size]]
        #zewnatrz trasowy
        # rest_points, start_solution, distance = find_best_point(rest_points, neighbors, index, distance)
        #wewnatrz
        start_solution, distance = find_best_swap(start_solution, index, distance, neighbors)
    print(distance)


#wewnatrz - krawedzie
def greedy_local_search_2(start_solution, rest_points, distance):
    choice = np.random.choice(2, cycle_size, replace=True)
    for index in range(cycle_size):
        neighbors = [start_solution[(index - 1) % cycle_size], start_solution[(index + 1) % cycle_size]]
        if choice[index] == 0:
            rest_points, start_solution, distance = find_first_better_point(rest_points, neighbors, start_solution[index], index, distance)
        else:
           start_solution, distance = find_first_better_edges(start_solution, index, distance)
    print(distance)


#wewnatrz - krawedzie
def steepest_local_search_2(start_solution, rest_points, distance):
    for index in range(cycle_size):
        neighbors = [start_solution[(index - 1) % cycle_size], start_solution[(index + 1) % cycle_size]]
        rest_points, start_solution, distance = find_best_point(rest_points, neighbors, index, distance)
        start_solution, distance = find_best_edges(start_solution, index, distance)
    print(distance)


cycle_size = round(int(np.ceil(len(problem.node_coords) / 2)))

start_solution = random.sample(range(len(problem.node_coords)), cycle_size)
rest_points = [x for x in range(100) if x not in start_solution]

adjacency_matrix = make_adjacency_matrix(problem.node_coords)
start_distance = get_path_length(adjacency_matrix, start_solution)
print( start_distance)

steepest_local_search_1(start_solution, rest_points, start_distance)
print(get_path_length(adjacency_matrix, start_solution))



# results_file = open(f'results_{instance_name}.csv', 'w')
# sys.stdout = results_file
# print('start_node', end=',')
# print('greedy', end=',')
# print('regret')
#
# greedy_results = []
# regret_results = []
# # start_nodes = random.sample(range(100), 10)
# for start_node in range(100):
#     print(start_node, end=',')
#     result_greedy = greedy_cycle(start_node)
#     greedy_length = get_path_length(adjacency_matrix, result_greedy)
#     greedy_results.append(greedy_length)
#     print(greedy_length, end=',')
#
#     result_regret = regret(start_node)
#     regret_length = get_path_length(adjacency_matrix, result_regret)
#     regret_results.append(regret_length)
#     print(regret_length)
#
#     if len(result_greedy) != cycle_size:
#         print("wrong cycle size of greedy result")
#         exit(-1)
#
#     if len(result_greedy) != cycle_size:
#         print("wrong cycle size of regret result")
#         exit(-1)
#
# print('MIN', end=',')
# print(min(greedy_results), end=',')
# print(min(regret_results))
# print('MAX', end=',')
# print(max(greedy_results), end=',')
# print(max(regret_results))
# print('AVG', end=',')
# print(np.mean(greedy_results), end=',')
# print(np.mean(regret_results))
#
# results_file.close()
