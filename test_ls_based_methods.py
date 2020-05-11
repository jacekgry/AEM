import sys
import time
from collections import defaultdict

import numpy as np

from aem import get_path_length, adjacency_matrix, problem, \
    instance_name, save_graph, MSLS, ILS1, ILS2, cycle_size

no_of_tests = 3
results = defaultdict(list)
times = defaultdict(list)
actual_distances = defaultdict(list)
start_distances = []
no_of_ls = defaultdict(list)
methods = [ILS2]

for method in methods:
    for i in range(no_of_tests):
        print(method.__name__)
        start_time = time.time()

        if method.__name__ == 'MSLS':
            solution, dist = method()
        else:
            # solution, dist = method(np.mean(times['MSLS']))
            solution, dist, ls = method(10.5)
            no_of_ls[method.__name__].append(ls)
            print('no of ls: ', ls)

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
print(f"AVG NO OF LS,{','.join(map(lambda x: str(np.mean(no_of_ls[x.__name__])), [ILS1, ILS2]))}")
print(f"MIN NO OF LS,{','.join(map(lambda x: str(np.min(no_of_ls[x.__name__])), [ILS1, ILS2]))}")
print(f"MAX NO OF LS,{','.join(map(lambda x: str(np.max(no_of_ls[x.__name__])), [ILS1, ILS2]))}")

print()
print('i,,,,,start_dist')
for i in range(no_of_tests):
    print(f"{i},{','.join(map(lambda x: str(results[x.__name__][i]), methods))}")
