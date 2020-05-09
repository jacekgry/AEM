import sys
import time
from collections import defaultdict

import numpy as np

from aem import get_path_length, adjacency_matrix, problem, \
    instance_name, save_graph, MSLS, ILS1, ILS2, cycle_size

no_of_tests = 2
results = defaultdict(list)
times = defaultdict(list)
actual_distances = defaultdict(list)
start_distances = []
methods = [MSLS, ILS1, ILS2]

for method in methods:
    for i in range(no_of_tests):
        print(method.__name__)
        start_time = time.time()

        if method.__name__ == 'MSLS':
            solution, dist = method()
        else:
            solution, dist = method(np.mean(times['MSLS']))

        print('cycle size: ', dist)

        assert len(solution) == cycle_size
        assert len(set(solution)) == len(solution)

        duration = time.time() - start_time
        print('duration: ', duration)
        results[method.__name__].append(dist)
        times[method.__name__].append(duration)
        actual_distances[method.__name__].append(get_path_length(adjacency_matrix, solution))
        print('actual size: ', get_path_length(adjacency_matrix, solution))

        save_graph(problem.node_coords, solution, f'{method.__name__}_{i}')

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
    print(f"{i},{','.join(map(lambda x: str(results[x.__name__][i]), methods))}")
