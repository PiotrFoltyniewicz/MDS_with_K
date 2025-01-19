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

def graph_from_file(filename):
    with open(filename) as f:
        n, m = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
        f.readline()
        k = int(f.readline().strip())
        
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
    for node in graph:
        node.nodes_in_range = get_nodes_in_range(node, k)
        node.cover_count = len(node.nodes_in_range)

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

def validate(graph, nodes, k):
    covered_nodes = set()

    for node in nodes:
        covered_nodes.update(get_nodes_in_range(graph[node.id], k))

    all_nodes = set(graph)
    return all_nodes.issubset(covered_nodes)

def heuristic_tester(solver):
    print("Test\tValid\tSize\tTime", sep='\t')
    for i in range(1, 11):
        start_time = time.time()
        graph, k = graph_from_file("heuristic-data/test" + str(i).zfill(2) + ".in")

        ans = solver(graph, k)

        elapsed_time = time.time() - start_time
        print("test" + str(i).zfill(2) , validate(graph, ans, k), len(ans), f"{elapsed_time:.2f}", sep='\t')