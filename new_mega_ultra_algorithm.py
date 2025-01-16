from utility import * 
from heapq import *

# with removal

def reduce_graph(graph, nodes, k):
   combined_to_remove = set()
   combined_boundary = set()

   for node in nodes:
      if node in graph:
         to_remove, boundary = get_nodes_in_range_plus_one(graph, node, k)
         combined_to_remove.update(to_remove)
         combined_boundary.update(boundary)

   for bound_node in combined_boundary:
      if bound_node in graph:
         graph[bound_node].difference_update(combined_to_remove)

   for node in combined_to_remove:
      graph.pop(node, None)

def initial_fill(graph, k):
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

      nodes_to_remove, boundary = get_nodes_in_range_plus_one(graph, best_node, k)  

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

def create_initial_solution(graph, k):
   init_sol = set()
   neighborhood_sizes = {}

   start_time = time.time()

   for node in graph:
      if time.time() - start_time > 20:
         init_sol.update(initial_fill(graph, k))
         return init_sol
      neighborhood_sizes[node] = len(get_nodes_in_range(graph, node, k))

   for node in list(graph.keys()):
      if len(graph[node]) == 0:
         init_sol.add(node)
      elif len(graph[node]) == 1:
         possible_nodes = get_nodes_in_range(graph, node, k)

         best = max(possible_nodes, key=lambda candidate: neighborhood_sizes[candidate])
         init_sol.add(best)
   reduce_graph(graph, init_sol, k)

   # fill the rest of the graph using greedy pick
   init_sol.update(initial_fill(graph, k))
         
   return init_sol

         

def algorithm(graph, k):
   best_solution = create_initial_solution(graph, k)

   return best_solution


heuristic_tester(algorithm)