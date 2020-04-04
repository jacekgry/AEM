import numpy as np
import sys
from aem import get_path_length, adjacency_matrix, cycle_size, instance_name, greedy_cycle, regret

results_file = open(f'results_{instance_name}.csv', 'w')
sys.stdout = results_file
print('start_node', end=',')
print('greedy', end=',')
print('regret')

greedy_results = []
regret_results = []
for start_node in range(100):
    print(start_node, end=',')
    result_greedy = greedy_cycle(start_node)
    greedy_length = get_path_length(adjacency_matrix, result_greedy)
    greedy_results.append(greedy_length)
    print(greedy_length, end=',')

    result_regret = regret(start_node)
    regret_length = get_path_length(adjacency_matrix, result_regret)
    regret_results.append(regret_length)
    print(regret_length)

    if len(result_greedy) != cycle_size:
        print("wrong cycle size of greedy result")
        exit(-1)

    if len(result_greedy) != cycle_size:
        print("wrong cycle size of regret result")
        exit(-1)

print('MIN', end=',')
print(min(greedy_results), end=',')
print(min(regret_results))
print('MAX', end=',')
print(max(greedy_results), end=',')
print(max(regret_results))
print('AVG', end=',')
print(np.mean(greedy_results), end=',')
print(np.mean(regret_results))

results_file.close()
