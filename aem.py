import itertools
import operator
import sys
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tsplib95 as tsp


def euclidean_distance(node1, node2):
    return int(((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5 + 0.5)


def get_nearest_vertices():
    result = []
    for row in adjacency_matrix:
        temp = np.argsort(row)[1:6]
        result.append(temp)
    return result


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


def find_combinations(matrix, order, node, prev_value):
    comb = []
    for z in range(1, len(order) + 1):
        combination = order[:]
        value = prev_value - adjacency_matrix[combination[z - 1]][combination[z % len(combination)]]
        combination.insert(z, node)
        value += adjacency_matrix[combination[z - 1]][node]
        value += adjacency_matrix[node][combination[(z + 1) % len(combination)]]
        comb.append((combination, value))
    return sorted(comb, key=operator.itemgetter(1))


def greedy_cycle_best_comb(matrix, order, node, value):
    comb = find_combinations(matrix, order, node, value)
    result = min(comb, key=operator.itemgetter(1))
    return result


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


def greedy_cycle_with_partial_solution(partial_solution):
    order_list = partial_solution
    dist = get_path_length(adjacency_matrix, order_list)
    for i in range(cycle_size - len(partial_solution)):
        temp = []
        for t in range(len(problem.node_coords)):
            if t not in order_list:
                temp.append(greedy_cycle_best_comb(adjacency_matrix, order_list, t, dist))
        order_list, dist = min(temp, key=operator.itemgetter(1))
    return order_list, dist


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
        current_unused_vertices = [x for x in range(no_of_nodes) if x not in current_sol]
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
        current_unused_vertices = [x for x in range(no_of_nodes) if x not in current_sol]
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
        current_unused_vertices = [x for x in range(no_of_nodes) if x not in current_sol]
    return current_sol, current_dist


def steepest_local_search_edges(start_sol, unused_vertices, start_dist):
    best_sol, best_distance = start_sol.copy(), start_dist
    current_unused_vertices = unused_vertices.copy()
    improved = True
    move = []
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
                    best_move = [delta, current_sol[index], current_sol[i], 0]
                    best_distance = current_dist + delta
                    improved = True

            for ext_idx, ext_vertex in enumerate(current_unused_vertices):
                neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
                if current_dist + delta < best_distance:
                    best_move = [delta, current_sol[index], ext_vertex, 1]
                    best_sol = current_sol.copy()
                    best_sol[index] = ext_vertex
                    best_distance = current_dist + delta
                    improved = True
        move.append(best_move)
        current_sol, current_dist = best_sol.copy(), best_distance
        current_unused_vertices = [x for x in range(no_of_nodes) if x not in current_sol]
    return current_sol, current_dist


def steepest_local_search_edges_with_ordered_move_list(start_sol, unused_vertices, start_dist):
    current_unused_vertices = unused_vertices.copy()
    improved = True
    current_sol = start_sol.copy()
    current_dist = start_dist
    move_list = []

    for index in range(cycle_size):
        for i in range(index + 1, cycle_size - 1):
            neighbors = [current_sol[(index - 1) % cycle_size], current_sol[(i + 1) % cycle_size]]
            delta = delta_edges(neighbors, current_sol[index], current_sol[i])
            if delta < 0:
                # internal
                move = [delta, [current_sol[index], current_sol[i]], 0]
                move_list.append(move)

        for ext_idx, ext_vertex in enumerate(current_unused_vertices):
            neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
            delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
            if delta < 0:
                # external
                move = [delta, [current_sol[index], ext_vertex], 1]
                move_list.append(move)
    best_move = None

    while improved:
        improved = False

        move_list = sorted(move_list, key=lambda move: move[0])
        new_move_list = move_list.copy()
        for move in move_list:
            internal = move[1][0]
            second_point = move[1][1]
            if move[2] and internal in current_sol and second_point not in current_sol:
                current_sol[current_sol.index(internal)] = second_point
                improved = True
                best_move = move
                break
            elif move[2] == 0 and internal in current_sol and second_point in current_sol:
                if (current_sol.index(second_point) + 1) % cycle_size == 0:
                    current_sol[current_sol.index(internal)::] = current_sol[current_sol.index(internal)::][::-1]
                else:
                    current_sol[current_sol.index(internal):current_sol.index(second_point) + 1] = current_sol[
                                                                                                   current_sol.index(
                                                                                                       internal):current_sol.index(
                                                                                                       second_point) + 1][
                                                                                                   ::-1]
                improved = True
                best_move = move
                break
            else:
                new_move_list.remove(move)
        # print(current_sol)
        # print(best_move)
        if improved:
            current_dist += best_move[0]
            indexes = []
            index = current_sol.index(best_move[1][1])
            indexes += [(index - 1) % cycle_size, index, (index + 1) % cycle_size]
            vert = [best_move[1][0]]
            if best_move[2] == 0:
                index = current_sol.index(best_move[1][0])
                indexes += [(index - 1) % cycle_size, index, (index + 1) % cycle_size]
            indexes = list(set(indexes))
            for index in indexes:
                vert.append(current_sol[index])
            new_move_list = move_list.copy()
            for i in move_list:
                if i[1][0] in vert or i[1][1] in vert:
                    new_move_list.remove(i)
            move_list = new_move_list
            for index in indexes:
                for i in range(cycle_size):
                    if index >= i or (current_sol[index] == current_sol[0] and current_sol[i] == current_sol[-1]):
                        continue
                    neighbors = [current_sol[(index - 1) % cycle_size], current_sol[(i + 1) % cycle_size]]
                    delta = delta_edges(neighbors, current_sol[index], current_sol[i])
                    move = [current_sol[index], current_sol[i]]
                    if delta < 0:
                        # internal
                        move_list.append([delta, move, 0])

                for ext_idx, ext_vertex in enumerate(current_unused_vertices):
                    neighbors1 = [current_sol[(index - 1) % cycle_size], current_sol[(index + 1) % cycle_size]]
                    delta = delta_of_swap(current_sol[index], ext_vertex, neighbors1)
                    move = [current_sol[index], ext_vertex]
                    if delta < 0:
                        # external
                        move_list.append([delta, move, 1])
            current_unused_vertices = [x for x in range(no_of_nodes) if x not in current_sol]
            vert.clear()
    return current_sol, get_path_length(adjacency_matrix, current_sol)


def steepest_local_search_edges_with_candidate_moves(start_sol, unused_vertices, start_dist):
    best_sol, best_distance = start_sol.copy(), start_dist
    current_unused_vertices = unused_vertices.copy()
    improved = True
    current_sol = start_sol.copy()
    current_dist = start_dist
    while improved:
        improved = False
        for index in range(cycle_size):
            for vert in nearest_vertices[current_sol[index]]:
                if vert in current_sol:
                    i = current_sol.index(vert)
                    if i > index and i != cycle_size - 1:
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
        current_unused_vertices = [x for x in range(no_of_nodes) if x not in current_sol]
    return current_sol, current_dist


def MSLS():
    result = []
    for i in range(100):
        new_start_solution = random.sample(range(len(problem.node_coords)), cycle_size)
        rest_points = [x for x in range(no_of_nodes) if x not in new_start_solution]
        start_distance = get_path_length(adjacency_matrix, new_start_solution)
        new_solution, new_dist = steepest_local_search_edges_with_ordered_move_list(new_start_solution, rest_points, start_distance)
        result.append([get_path_length(adjacency_matrix, new_solution), new_solution])
    best_sol = sorted(result, key=lambda x: x[0])[0]
    return best_sol[1], best_sol[0]


def ILS1(avg_mlsl_time, no_of_random_swaps=5):
    no_of_ls = 1
    start_time = time.time()
    new_start_solution = random.sample(range(len(problem.node_coords)), cycle_size)
    rest_points = [x for x in range(no_of_nodes) if x not in new_start_solution]
    start_distance = get_path_length(adjacency_matrix, new_start_solution)
    print("start distance: ", start_distance)
    solution, dist = steepest_local_search_edges_with_ordered_move_list(new_start_solution, rest_points, start_distance)
    while time.time() - start_time < avg_mlsl_time:
        no_of_ls += 1
        vertices_to_delete = random.sample(solution, no_of_random_swaps)
        vertices_to_add = random.sample(rest_points, no_of_random_swaps)
        change = 0
        solution_after_perturbation = solution.copy()
        rest_points_after_perturbation = rest_points.copy()
        for vertex_to_delete, vertex_to_add in zip(vertices_to_delete, vertices_to_add):
            idx_of_vertex_to_delete = solution_after_perturbation.index(vertex_to_delete)
            neighbors1 = [solution_after_perturbation[(idx_of_vertex_to_delete - 1) % cycle_size],
                          solution_after_perturbation[(idx_of_vertex_to_delete + 1) % cycle_size]]
            change += delta_of_swap(vertex_to_delete, vertex_to_add, neighbors1)
            solution_after_perturbation[idx_of_vertex_to_delete] = vertex_to_add
            rest_points_after_perturbation.append(vertex_to_delete)
            rest_points_after_perturbation.remove(vertex_to_add)

        solution_after_ls, dist_after_ls = steepest_local_search_edges_with_ordered_move_list(solution_after_perturbation, rest_points_after_perturbation,
                                                                                              dist + change)
        if dist_after_ls < dist:
            solution = solution_after_ls
            dist = dist_after_ls
            rest_points = [x for x in range(no_of_nodes) if x not in solution_after_ls]
    return solution, dist, no_of_ls


def ILS2(avg_mlsl_time, percentage_of_deleted_solution=0.2):
    no_of_ls = 1
    start_time = time.time()
    new_start_solution = random.sample(range(len(problem.node_coords)), cycle_size)
    rest_points = [x for x in range(no_of_nodes) if x not in new_start_solution]
    start_distance = get_path_length(adjacency_matrix, new_start_solution)
    solution, dist = steepest_local_search_edges_with_ordered_move_list(new_start_solution, rest_points, start_distance)
    while time.time() - start_time < avg_mlsl_time:
        no_of_ls += 1
        first_vertex_idx_to_delete = random.randrange(0, cycle_size - int(percentage_of_deleted_solution * cycle_size))
        to_be_deleted_from_beggining = first_vertex_idx_to_delete + int(percentage_of_deleted_solution * cycle_size)
        solution_after_perturbation = solution[:first_vertex_idx_to_delete]
        solution_after_perturbation += solution[to_be_deleted_from_beggining:]
        solution_after_greedy, dist_after_greedy = greedy_cycle_with_partial_solution(solution_after_perturbation)
        solution_after_ls, dist_after_ls = steepest_local_search_edges_with_ordered_move_list(solution_after_greedy,
                                                                                              [x for x in range(no_of_nodes) if x not in solution_after_greedy],
                                                                                              dist_after_greedy)
        if dist_after_ls < dist:
            solution = solution_after_ls
            dist = dist_after_ls

    return solution, dist, no_of_ls


instance_name = sys.argv[1] if len(sys.argv) == 2 else 'kroB200'
problem: tsp.Problem = tsp.load_problem(f'{instance_name}.tsp')
cycle_size = round(int(np.ceil(len(problem.node_coords) / 2)))
# cycle_size = 30
adjacency_matrix = make_adjacency_matrix(problem.node_coords)
no_of_nodes = len(adjacency_matrix)
nearest_vertices = get_nearest_vertices()
