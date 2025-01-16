import random
from collections import deque
import time

# Function to read the graph from input
def read_graph():
    n, m = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(m)]
    input()
    k = int(input())

    graph = {i: set() for i in range(n)}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    return graph, n, k

# Get nodes covered by a node within distance k using BFS
def get_k_hop_neighbors(graph, node, k):
    visited = set()
    queue = deque([(node, 0)])
    covered_nodes = set()

    while queue:
        current, dist = queue.popleft()
        if dist > k:
            break
        if current not in visited:
            visited.add(current)
            covered_nodes.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
    
    return covered_nodes

# Initialize random population
def initialize_population(num_nodes, pop_size):
    return [random.choices([0, 1], k=num_nodes) for _ in range(pop_size)]

# Fitness function: maximize coverage and minimize set size
def fitness(solution, graph, k, num_nodes):
    covered_nodes = set()
    for i, selected in enumerate(solution):
        if selected:
            covered_nodes.update(get_k_hop_neighbors(graph, i, k))

    coverage = len(covered_nodes) / num_nodes
    size = sum(solution)

    if coverage < 1.0:  # Apply penalty for incomplete coverage
        return -float('inf')

    return coverage - 0.01 * size  # Penalize larger sets

# Crossover between two parents
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

# Mutation: flip a random bit
def mutate(solution):
    idx = random.randint(0, len(solution) - 1)
    solution[idx] = 1 - solution[idx]
    return solution

# Select parents using tournament selection
def select_parents(population, fitness_scores):
    tournament_size = 3
    candidates = random.sample(list(zip(population, fitness_scores)), tournament_size)
    return max(candidates, key=lambda x: x[1])[0]

# Genetic algorithm main function
def genetic_algorithm(graph, num_nodes, k, pop_size=50, mutation_rate=0.1):
    population = initialize_population(num_nodes, pop_size)
    start_time = time.time()

    gen = 1
    while time.time() - start_time < 25:
        print("Generation", gen)
        gen += 1
        fitness_scores = [fitness(individual, graph, k, num_nodes) for individual in population]
        population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]

        next_generation = population[:5]  # Elitism: retain top 5 solutions

        while len(next_generation) < pop_size:
            parent1 = select_parents(population, fitness_scores)
            parent2 = select_parents(population, fitness_scores)
            child = mutate(crossover(parent1, parent2)) if random.random() < mutation_rate else crossover(parent1, parent2)
            next_generation.append(child)

        # Stop if convergence criteria met (optional)
        if max(fitness_scores) == 1:
            break

        population = next_generation

    # Best solution from final population
    best_solution = max(population, key=lambda s: fitness(s, graph, k, num_nodes))
    return [i for i, val in enumerate(best_solution) if val == 1]


def get_nodes_in_range(graph, start_node, k):
    nodes_within_range = set()
    visited = set()
    queue = deque([(start_node, 0)])
    
    while queue:
        current_node, depth = queue.popleft()
        
        if depth <= k:
            nodes_within_range.add(current_node)
            
            if current_node not in visited:
                visited.add(current_node)
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
    
    return nodes_within_range

def validate(graph, nodes, k):
    covered_nodes = set()

    for node in nodes:
        covered_nodes.update(get_nodes_in_range(graph, node, k))

    all_nodes = set(graph.keys())
    return all_nodes.issubset(covered_nodes)

graph, num_nodes, k = read_graph()
result = genetic_algorithm(graph, num_nodes, k)
print("Minimum k-hop dominating set:", len(result))
print(validate(graph, result, k))