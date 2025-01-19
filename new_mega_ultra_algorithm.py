from utility import * 
from heapq import *
import random

class HeapNode():
    def __init__(self, original):
        self.original = original
    
    def __lt__(self, other):
        return len(self.original.get_nodes_in_range()) > len(other.original.get_nodes_in_range())

    def __eq__(self, other):
        return self.original == other.original
    
def reduce_graph(graph, del_node):
    for neighbor in del_node.get_nodes_in_range():
        neighbor.is_covered = True
        graph.remove(neighbor)
    
def greedy_pick(graph):
    solution = set()
    pq = [HeapNode(node) for node in graph]
    heapify(pq)
   
    while graph and pq:
        best_node = heappop(pq).original
        if best_node not in graph:
            continue
        solution.add(best_node)
        reduce_graph(graph, best_node)
    
    return solution

def leaf_reduction(graph):
    init_sol = set()
    i = 0
    while i < len(graph):
        node = graph[i]
        if len(node.get_nearest()) == 0:
            init_sol.add(node)
            i += 1
        elif len(node.get_nearest()) == 1:
            best = max(node.nodes_in_range)
            init_sol.add(best)
            reduce_graph(graph, best)
        else:
            i += 1
         
    return init_sol

def greedy_deconstruction(graph, nodes, percentage = 0.25):
    temp_nodes = sorted(list(nodes), key= lambda node: node.cover_count)
    first_idx = int(len(nodes) * percentage)
    temp_nodes = temp_nodes[first_idx:]
    temp_graph = clone_graph(graph)
    temp_nodes = [temp_graph[node.id] for node in temp_nodes]

    for node in temp_nodes:
        if node in temp_graph:
            reduce_graph(temp_graph, node)

    return temp_graph, set(temp_nodes)

def random_deconstruction(graph, nodes, percentage = 0.2):
    temp_graph = clone_graph(graph)
    temp_nodes = [temp_graph[node.id] for node in nodes if random.random() < percentage]

    for node in temp_nodes:
        if node in temp_graph:
            reduce_graph(temp_graph, node)

    return temp_graph, set(temp_nodes)


def algorithm(original_graph, k):
    start_time = time.time()
    best_solution = set()
    extend_graph(original_graph, k)
    graph = clone_graph(original_graph)


    best_solution.update(leaf_reduction(graph))
    best_solution.update(greedy_pick(graph))

    while time.time() - start_time < 20:
        graph, nodes  = greedy_deconstruction(original_graph, best_solution)
        nodes.update(greedy_pick(graph))
        if len(nodes) < len(best_solution):
            best_solution = nodes

        graph, nodes  = random_deconstruction(original_graph, best_solution)
        nodes.update(greedy_pick(graph))
        if len(nodes) < len(best_solution):
            best_solution = nodes

    return best_solution


heuristic_tester(algorithm)