from typing import Dict, List, Set
from collections import deque
import sys
import random
from typing import Dict, Set, Callable
from copy import deepcopy
from heapq import heappush, heappop
import time

def graph_from_input():
    n, m = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(m)]
    input()
    k = int(input())

    graph = {i: set() for i in range(n)}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    return graph, k
        
def get_nodes_in_range_with_boundary(graph: Dict[int, Set[int]], start_node: int, k: int) -> tuple[Set[int], Set[int]]:
    nodes_within_range = set()
    boundary_nodes = set()
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
                    if depth + 1 > k:
                        boundary_nodes.add(neighbor)
    
    for node in nodes_within_range:
        for neighbor in graph[node]:
            if neighbor not in nodes_within_range:
                boundary_nodes.add(neighbor)
    
    return nodes_within_range, boundary_nodes

def get_nodes_in_range(graph: Dict[int, Set[int]], start_node: int, k: int) -> Set[int]:
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

def remove_neighborhood(graph: Dict[int, Set[int]], node: int, k: int) -> None:
    to_remove, boundary = get_nodes_in_range_with_boundary(graph, node, k)
    
    for bound_node in boundary:
        if bound_node in graph:
            graph[bound_node] -= to_remove
    
    for node in to_remove:
        if node in graph:
            del graph[node]

def reduce_graph(graph: Dict[int, Set[int]], nodes: Set[int], k: int):
    for node in nodes:
        if node in graph:
            remove_neighborhood(graph, node, k)


def greedy_pick(graph: Dict[int, Set[int]], k: int) -> Set[int]:
    output = set()
    
    pq = []
    degree_map = {}
    
    def update_priority_queue() -> None:
        pq.clear()
        for node in graph:
            degree = len(get_nodes_in_range(graph, node, k))
            degree_map[node] = degree
            heappush(pq, (-degree, node))
    
    def remove_neighborhood(node: int) -> None:

        nodes_to_remove, boundary = get_nodes_in_range_with_boundary(graph, best_node, k)  

        for bound_node in boundary:
            if bound_node in graph:
                graph[bound_node] -= nodes_to_remove
                degree_map[bound_node] = len(graph[bound_node])
                heappush(pq, (-degree_map[bound_node], bound_node))
        
        for node in nodes_to_remove:
            if node in graph:
                del graph[node]
                degree_map.pop(node, None)
    
    update_priority_queue()
    
    while graph:
        while pq and -pq[0][0] != degree_map.get(pq[0][1], 0):
            heappop(pq)
            
        if not pq:
            update_priority_queue()
            if not pq:
                break
                
        _, best_node = heappop(pq)
        
        if best_node not in graph:
            continue
            
        output.add(best_node)
        remove_neighborhood(best_node)
    
    return output

def local_search(graph: Dict[int, Set[int]], k: int, solver: Callable[[Dict[int, Set[int]], int], Set[int]]) -> Set[int]:
    start_time = time.time()
    temp_graph = deepcopy(graph)
    solve_time = time.time()
    best_solution = solver(temp_graph, k)
    solve_time = time.time() - solve_time
    best_len = len(best_solution)
    elapsed_time = 0
    
    while elapsed_time < 25 and solve_time < 13:

        temp_graph = deepcopy(graph)
        solution_copy = list(best_solution.copy())[1:]
        reduce_graph(temp_graph, solution_copy, k)
        solution_copy.update(solver(temp_graph, k))

        if len(solution_copy) < best_len:
            best_solution = solution_copy
            best_len = len(solution_copy)

        elapsed_time = time.time() - start_time

    return best_solution

graph, k = graph_from_input()
ans = local_search(graph, k, greedy_pick)

print(len(ans))

for n in ans:
    print(n, end=' ')
