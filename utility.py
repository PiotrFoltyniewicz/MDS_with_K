import sys
import time
from copy import deepcopy
from collections import deque, defaultdict

def graph_from_input():
   n, m = map(int, sys.stdin.readline().split())
   edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(m)]
   sys.stdin.readline()
   k = int(sys.stdin.readline())

   graph = {i: set() for i in range(n)}
   for u, v in edges:
      graph[u].add(v)
      graph[v].add(u)
   return graph, k

def graph_from_file(filename):
    with open(filename) as f:
        n, m = map(int, f.readline().split())
        edges = [tuple(map(int, f.readline().split())) for _ in range(m)]
        f.readline()
        k = int(f.readline().strip())
        
    graph = {i: set() for i in range(n)}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    return graph, k

def extend_graph(graph, k):
    n = len(graph)
    extended_graph = defaultdict(set)
    
    for node in range(n):
        nodes_in_range = get_nodes_in_range(graph, node, k)
        extended_graph[node].update(nodes_in_range - {node})

    graph = dict(extended_graph)

def get_nodes_in_range(graph, start_node, k):
    nodes_within_range = set()
    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, depth = queue.popleft()

        if depth > k:
            continue

        nodes_within_range.add(current_node)

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return nodes_within_range

def get_nodes_in_range_plus_one(graph, start_node, k):
    nodes_within_k = set()         # Nodes within range k
    nodes_within_k_plus_one = set()  # Nodes within range k + 1
    queue = deque([(start_node, 0)])
    visited = {start_node}

    while queue:
        current_node, depth = queue.popleft()

        # Add nodes to respective sets based on depth
        if depth <= k:
            nodes_within_k.add(current_node)
        elif depth == k + 1:
            nodes_within_k_plus_one.add(current_node)
        else:
            continue

        # Continue BFS for neighbors
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return nodes_within_k, nodes_within_k_plus_one

def validate(graph, nodes, k):
    covered_nodes = set()

    for node in nodes:
        covered_nodes.update(get_nodes_in_range(graph, node, k))

    all_nodes = set(graph.keys())
    return all_nodes.issubset(covered_nodes)

def heuristic_tester(solver):
   print("Test\tValid\tSize\tTime", sep='\t')
   for i in range(1, 11):
      start_time = time.time()
      graph, k = graph_from_file("heuristic-data/test" + str(i).zfill(2) + ".in")
      temp_graph = deepcopy(graph)
      ans = solver(temp_graph, k)

      elapsed_time = time.time() - start_time
      print("test" + str(i).zfill(2) , validate(graph, ans, k), len(ans), f"{elapsed_time:.2f}", sep='\t')