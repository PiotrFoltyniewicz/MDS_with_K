from heapq import *
import random
import sys
import time
from copy import *
from collections import deque

class Node:
    def __init__(self, id):
        self.id = id
        self.nearest_neighbors = set()
        self.nodes_in_range = set()
        self.is_covered = False
        self.cover_count = 0

    def get_nearest(self):
        self.nearest_neighbors = {node for node in self.nearest_neighbors if not node.is_covered}
        return self.nearest_neighbors
    
    def get_nodes_in_range(self):
        self.nodes_in_range = {node for node in self.nodes_in_range if not node.is_covered}
        return self.nodes_in_range
    
    def __lt__(self, other):
        return len(self.get_nodes_in_range()) < len(other.get_nodes_in_range())

def graph_from_input():
    n, m = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(m)]
    sys.stdin.readline()
    k = int(sys.stdin.readline())

    graph = [Node(i) for i in range(n)]
    for u, v in edges:
       graph[u].nearest_neighbors.add(graph[v])
       graph[v].nearest_neighbors.add(graph[u])
    return graph, k

def clone_graph(nodes):
    node_mapping = {}
    
    for original_node in nodes:
        new_node = Node(original_node.id)
        new_node.is_covered = original_node.is_covered
        new_node.cover_count = original_node.cover_count
        node_mapping[original_node] = new_node
    
    for original_node in nodes:
        new_node = node_mapping[original_node]
        
        new_node.nearest_neighbors = {
            node_mapping[neighbor] 
            for neighbor in original_node.nearest_neighbors
            if neighbor in node_mapping
        }
        
        new_node.nodes_in_range = {
            node_mapping[node] 
            for node in original_node.nodes_in_range
            if node in node_mapping
        }
    
    return list(node_mapping.values())

def extend_graph(graph, k):
    good_nodes = set()
    for node in graph:
        node.nodes_in_range = get_nodes_in_range(node, k)
        node.cover_count = len(node.nodes_in_range)
        if node.cover_count > 0.1 * len(graph):
            good_nodes.add(node)

    for node in good_nodes:
        if node in graph:
            reduce_graph(graph, node)
    return good_nodes

def get_nodes_in_range(start_node: Node, k):
    nodes_within_range = set()
    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, depth = queue.popleft()

        if depth > k:
            continue

        nodes_within_range.add(current_node)
        for neighbor in current_node.nearest_neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return nodes_within_range

def reduce_graph(graph, del_node):
    for neighbor in del_node.get_nodes_in_range():
        neighbor.is_covered = True
        graph.remove(neighbor)


class HeapNode():
    def __init__(self, original):
        self.original = original
    
    def __lt__(self, other):
        return len(self.original.get_nodes_in_range()) > len(other.original.get_nodes_in_range())

    def __eq__(self, other):
        return self.original == other.original
    
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
    best_solution.update(extend_graph(original_graph, k))
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


graph, k = graph_from_input()
solution = algorithm(graph, k)
print(len(solution))
for node in solution:
    print(node.id, end=' ')