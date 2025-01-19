import time
import sys
import random

class Node:
    def __init__(self, id):
        self.id = id
        self.is_covered = False
        self.neighbors = []

start_time = 0
terminate_flag = False

# get number of uncovered nodes in range k and also list of nodes in range
def get_nodes_in_range(start_node, k):
    output = []
    queue = [(start_node, 0)]
    uncovered = 0
    visited = set()

    while queue:
        node, dist = queue.pop(0)
        if not node.is_covered:
            uncovered += 1

        if dist == k:
            output.append(node)
            visited.add(node.id)
            if not node.is_covered:
                uncovered += 1

        elif dist < k:
            for neighbor in node.neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                output.append(node)
                queue.append((neighbor, dist + 1))

    return output, uncovered

#  mark all nodes in range k as covered
def cover_nodes(start_node, k):
    queue = [(start_node, 0)]
    visited = set()

    while queue:
        node, dist = queue.pop(0)
        node.is_covered = True

        if dist == k:
            visited.add(node.id)
            node.is_covered = True

        elif dist < k:
            for neighbor in node.neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

def leaf_reduction(graph, k):
    global start_time
    global terminate_flag
    initial_solution = []
    for node in graph:
        if time.time() - start_time >= 27:
            terminate_flag = True
            return [False for _ in range(len(graph))]
        if node.is_covered:
            continue
        if len(node.neighbors) == 0:
            node.is_covered = True
            initial_solution.append(node)
        # reduce 1 degree nodes
        # TLDR - if we have leaf node, then we look to find the best node in range k, and it will be in the solution
        # in case of k = 1, it works perfectly, i am not sure if it is the most optimal for k > 1
        elif len(node.neighbors) == 1:
            possible_nodes, _ = get_nodes_in_range(node, k)
            best_node, highest_cover = possible_nodes[0], -1

            # my mind is gone, so let's say that is a medieval tournament
            for knight in possible_nodes:
                _, cover_count = get_nodes_in_range(knight, k)
                if cover_count > highest_cover:
                    best_node = knight
                    highest_cover = cover_count
            initial_solution.append(best_node)
            cover_nodes(best_node, k)
    return initial_solution
            
def greedy_pick(graph, k):
    global start_time
    global terminate_flag

    solution = []
    for node in graph:
        if time.time() - start_time >= 27:
            terminate_flag = True
            return [False for _ in range(len(graph))]
        if node.is_covered:
            continue

        possible_nodes, node_cover = get_nodes_in_range(node, k)
        best_node, highest_cover = node, node_cover

        for pos_node in possible_nodes:
            _, cover_count = get_nodes_in_range(pos_node, k)
            if cover_count > highest_cover:
                best_node = pos_node
                highest_cover = cover_count
        solution.append(best_node)
        cover_nodes(best_node, k)
    return solution

def greedy_solve(graph, k):
    global start_time
    global terminate_flag
    solution = []
    solution.extend(leaf_reduction(graph, k))

    if terminate_flag:
        return [False for _ in range(len(graph))]
    
    solution.extend(greedy_pick(graph, min(2, k)))
    
    return solution

def genetic_optimization(graph, best_solution, k):
    global start_time
    global terminate_flag
    
    def mutate(solution):
        mutated = solution.copy()
        index = random.randint(0, len(mutated) - 1)
        mutated[index] = not mutated[index]
        return mutated
    
    def crossover(solution1, solution2):
        child = solution1[:len(solution1) // 2] + solution2[len(solution2) // 2:]
        child = mutate(child)
        return child
    
    def convert_to_normal(genetic_solution):
        global terminate_flag
        if time.time() - start_time >= 27:
            terminate_flag = True
        return [graph[i] for i in range(len(genetic_solution)) if genetic_solution[i]]
    
    converted_sol = []
    for node in graph:
        converted_sol.append(True if node in best_solution else False)
        if time.time() - start_time >= 27:
            terminate_flag = True
            return best_solution
    population_size = 100

    # initialize population based on initial solution
    population = [converted_sol] + [mutate(converted_sol) for _ in range(population_size - 1)]
    normal_pop = []
    for p in population:
        if terminate_flag:
            break
        normal_pop.append(deep_validate(graph, convert_to_normal(p), k))
    if not terminate_flag:
        population = sorted(population, key=lambda x: x.count(True) if normal_pop[population.index(x)] else float('inf'))

    while not terminate_flag:
        # evaluation
        # selecton
        population = population[:population_size // 2]
        temp_size = len(population) - 1
        # filling the population back
        while not terminate_flag and len(population) < population_size:
            population.append(crossover(population[random.randint(0, temp_size)], population[random.randint(0, temp_size)]))

        normal_pop = []
        for p in population:
            if terminate_flag:
                break
            normal_pop.append(deep_validate(graph, convert_to_normal(p), k))
        if not terminate_flag:
            population = sorted(population, key=lambda x: x.count(True) if normal_pop[population.index(x)] else float('inf'))
    return convert_to_normal(population[0])


# it is better to solve big graph unoptimally than get TLE
def ramp_up_runner(graph, k):
    global start_time
    global terminate_flag
    
    current_k = min(2, k)
    solution = greedy_solve(graph, current_k)

    while not terminate_flag and current_k < k:
        current_k += 1
        # graph reset
        for node in graph:
            node.is_covered = False
        alternative = greedy_solve(graph, current_k)

        if len(alternative) < len(solution):
            solution = alternative
    if not terminate_flag:
        solution = genetic_optimization(graph, solution, k)
    return solution

def optilio_run():
    global start_time
    start_time = time.time()
    graph, k = graph_from_input()
    solution = ramp_up_runner(graph, k)

    print(len(solution))

    for node in solution:
        print(node.id, end=' ')

# validates based on a solution
def deep_validate(graph, solution, k):
    for node in graph:
        node.is_covered = False
    for node in solution:
        cover_nodes(node, k)

    return all([node.is_covered for node in graph])

def graph_from_input():
    n, m = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(m)]
    sys.stdin.readline()
    k = int(sys.stdin.readline().strip())
        
    graph = [Node(i) for i in range(n)]
    for u, v in edges:
       graph[u].neighbors.append(graph[v])
       graph[v].neighbors.append(graph[u])
    return graph, k

optilio_run()