## Minimum dominating set problem with K

### Description
This algorithm is an solution to a optimization problem, which can be seen here https://www.optil.io/optilion/problem/3216 <br>
Under vague description hides minimum dominating set problem with addition, that each node in solution covers all nodes in range K (unlike in classical problem where each node in solution only covers it's neighbors).<br>
<br>
Solving this problem was an individual project for university Combinatorial Optimization classes.

### Approach explanation
To solve this optimization problem I used GRASP approach.<br>
This is how step by step it works.
1. Firstly, program runs an very **quick greedy algorithm**, which doesn't give optimal solution, however at least it gives solution for very big graphs.
2. Then we run **reduction** function to mark add unconnected nodes to the solution and also find optimal nodes which cover leaf nodes.
3. With graph reduced, we run more optimal but still **greedy algorithm**, which almost surely will give better result than quick greedy algorithm.
4. If a solution was found within time limit (30 seconds), we run local search:
    - **Local improvement** - removing one node from solution and replacing it with another node which doesn't violate solution's validity
    - **Deconstruction** - randomly removing part of the solution (by default 20%)
    - **Reconstruction** - running greedy algorithm from point 3. to fill the gaps in the solution
5. This local search runs until time limit is achieved, also it is worth noting, that program frequently checks if the time limit is achieved, if so, then program returns the best solution found so far.
(it might happen that even only reduction won't have time to finish in time limit for the biggest graphs)
<br>
At the moment of writing this README.md file, this solution is the best one on the website, where problem can be found, so I conclude it is pretty effective.
