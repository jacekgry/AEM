import random
import sys
import time
from collections import defaultdict

import numpy as np

from aem import steepest_local_search_edges, steepest_local_search_vertices, \
    greedy_local_search_edges, greedy_local_search_vertices, get_path_length, adjacency_matrix, cycle_size, problem, \
    instance_name, save_graph, steepest_local_search_edges_with_ordered_move_list,steepest_local_search_edges_with_candidate_moves

no_of_tests = 10
results = defaultdict(list)
times = defaultdict(list)
actual_distances = defaultdict(list)
start_distances = []
methods = [steepest_local_search_edges, steepest_local_search_edges_with_ordered_move_list, steepest_local_search_edges_with_candidate_moves]


for i in range(no_of_tests):
    start_solution = random.sample(range(len(problem.node_coords)), cycle_size)
    # start_solution = greedy_cycle(i)
    rest_points = [x for x in range(100) if x not in start_solution]
    start_distance = get_path_length(adjacency_matrix, start_solution)
    start_distances.append(start_distance)
    for method in methods:
        s, r = start_solution.copy(), rest_points.copy()
        start_time = time.time()
        solution, dist = method(s, r, start_distance)
        duration = time.time() - start_time
        results[method.__name__].append(dist)
        times[method.__name__].append(duration)
        actual_distances[method.__name__].append(get_path_length(adjacency_matrix, solution))
        # save_graph(problem.node_coords, solution, f'{method.__name__}_{i}')

results_file = open(f'results_ls_{instance_name}.csv', 'w')
sys.stdout = results_file
print(f",{','.join(map(lambda x: x.__name__, methods))}")

print(f"AVG,{','.join(map(lambda x: str(np.mean(results[x.__name__])), methods))}")
print(f"MIN,{','.join(map(lambda x: str(np.min(results[x.__name__])), methods))}")
print(f"MAX,{','.join(map(lambda x: str(np.max(results[x.__name__])), methods))}")
print(f"AVG TIME,{','.join(map(lambda x: str(np.mean(times[x.__name__])), methods))}")
print()
print('i,,,,,start_dist')
for i in range(no_of_tests):
    print(f"{i},{','.join(map(lambda x: str(results[x.__name__][i]), methods))},{start_distances[i]}")
