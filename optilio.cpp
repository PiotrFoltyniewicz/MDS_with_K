#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>

class Node
{
public:
    int id;
    bool is_covered;
    std::vector<Node *> neighbors;

    Node(int id) : id(id), is_covered(false) {}
};

double start_time;
bool terminate_flag = false;

// bfs which gets list of outer nodes in range k and also number of nodes covered by start_node
// such approach gives worse results, than full bfs, however in later calculations it is much faster
std::pair<std::vector<Node *>, int> get_outer_in_range(Node *start_node, int k)
{
    std::vector<Node *> output;
    std::queue<std::pair<Node *, int>> q;
    int uncovered = 0;
    std::unordered_set<int> visited;

    q.push({start_node, 0});

    while (!q.empty())
    {
        auto [node, dist] = q.front();
        q.pop();

        if (!node->is_covered)
        {
            uncovered++;
        }

        if (dist == k)
        {
            output.push_back(node);
            visited.insert(node->id);
            if (!node->is_covered)
            {
                uncovered++;
            }
        }
        else if (dist < k)
        {
            for (Node *neighbor : node->neighbors)
            {
                if (visited.find(neighbor->id) != visited.end())
                {
                    continue;
                }
                visited.insert(neighbor->id);
                q.push({neighbor, dist + 1});
            }
        }
    }

    return {output, uncovered};
}

// marks all nodes in range k as covered
void cover_nodes(Node *start_node, int k)
{
    std::queue<std::pair<Node *, int>> q;
    std::unordered_set<int> visited;

    q.push({start_node, 0});

    while (!q.empty())
    {
        auto [node, dist] = q.front();
        q.pop();
        node->is_covered = true;

        if (dist == k)
        {
            visited.insert(node->id);
            node->is_covered = true;
        }
        else if (dist < k)
        {
            for (Node *neighbor : node->neighbors)
            {
                if (visited.find(neighbor->id) != visited.end())
                {
                    continue;
                }
                visited.insert(neighbor->id);
                q.push({neighbor, dist + 1});
            }
        }
    }
}

// automatically adds nodes without neighbors to the solution
// then for each nodes with one neighbor (leaf node)
// calculates the best node from range k to add to solution

std::vector<Node *> leaf_reduction(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> initial_solution;

    for (Node *node : graph)
    {
        if ((std::chrono::steady_clock::now().time_since_epoch().count() / 1e9) - start_time >= 27)
        {
            terminate_flag = true;
            return graph;
        }

        if (node->is_covered)
            continue;

        if (node->neighbors.empty())
        {
            node->is_covered = true;
            initial_solution.push_back(node);
        }
        else if (node->neighbors.size() == 1)
        {
            auto [possible_nodes, dummy_cover_count] = get_outer_in_range(node, k);

            // initially we add the only neighbor of leaf node
            Node *best_node = node->neighbors[0];
            auto [dummy_node, current_cover] = get_outer_in_range(best_node, k);
            int highest_cover = current_cover;

            // my mind is gone, so let's say this is a medieval tournament
            for (Node *knight : possible_nodes)
            {
                auto [_, cover_count] = get_outer_in_range(knight, k);
                if (cover_count > highest_cover)
                {
                    best_node = knight;
                    highest_cover = cover_count;
                }
            }
            initial_solution.push_back(best_node);
            cover_nodes(best_node, k);
        }
    }
    return initial_solution;
}

// goes through the graph and adds nodes to the solution based on greedy approach
std::vector<Node *> greedy_pick(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> solution;

    for (Node *node : graph)
    {
        if ((std::chrono::steady_clock::now().time_since_epoch().count() / 1e9) - start_time >= 27)
        {
            terminate_flag = true;
            return graph;
        }

        if (node->is_covered)
            continue;

        auto [possible_nodes, node_cover] = get_outer_in_range(node, k);
        Node *best_node = node;
        int highest_cover = node_cover;

        for (Node *pos_node : possible_nodes)
        {
            auto [_, cover_count] = get_outer_in_range(pos_node, k);
            if (cover_count > highest_cover)
            {
                best_node = pos_node;
                highest_cover = cover_count;
            }
        }
        solution.push_back(best_node);
        cover_nodes(best_node, k);
    }
    return solution;
}

// combines leaf reduction and greedy pick into one function
// also checks if program shouldn't be stopped early (needed for big graphs)
std::vector<Node *> greedy_solve(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> solution;
    auto leaf_solution = leaf_reduction(graph, k);
    solution.insert(solution.end(), leaf_solution.begin(), leaf_solution.end());

    if (terminate_flag)
    {
        return graph;
    }

    auto greedy_solution = greedy_pick(graph, std::min(2, k));
    solution.insert(solution.end(), greedy_solution.begin(), greedy_solution.end());

    return solution;
}

// validates solution based on solution
bool deep_validate(std::vector<Node *> &graph, const std::vector<Node *> &solution, int k)
{
    for (Node *node : graph)
    {
        node->is_covered = false;
    }
    for (Node *node : solution)
    {
        cover_nodes(node, k);
    }

    return std::all_of(graph.begin(), graph.end(), [](Node *node)
                       { return node->is_covered; });
}

// genetic optimization based on binary array
std::vector<Node *> genetic_optimization(std::vector<Node *> &graph, const std::vector<Node *> &best_solution, int k)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    // randomly changes one node
    auto mutate = [&](const std::vector<bool> &solution)
    {
        std::vector<bool> mutated = solution;
        std::uniform_int_distribution<> dis(0, mutated.size() - 1);
        int index = dis(gen);
        mutated[index] = !mutated[index];
        return mutated;
    };

    // combines first part of solution1 with second part of solution2
    auto crossover = [&](const std::vector<bool> &solution1, const std::vector<bool> &solution2)
    {
        std::vector<bool> child;
        int mid = solution1.size() / 2;
        child.insert(child.end(), solution1.begin(), solution1.begin() + mid);
        child.insert(child.end(), solution2.begin() + mid, solution2.end());
        return mutate(child);
    };

    // converts binary solution into normal one
    auto convert_to_normal = [&](const std::vector<bool> &genetic_solution)
    {
        if ((std::chrono::steady_clock::now().time_since_epoch().count() / 1e9) - start_time >= 27)
        {
            terminate_flag = true;
        }
        std::vector<Node *> result;
        for (size_t i = 0; i < genetic_solution.size(); ++i)
        {
            if (genetic_solution[i])
            {
                result.push_back(graph[i]);
            }
        }
        return result;
    };

    std::vector<bool> converted_sol(graph.size(), false);
    for (size_t i = 0; i < graph.size(); ++i)
    {
        if ((std::chrono::steady_clock::now().time_since_epoch().count() / 1e9) - start_time >= 27)
        {
            terminate_flag = true;
            return best_solution;
        }

        converted_sol[i] = std::find(best_solution.begin(), best_solution.end(), graph[i]) != best_solution.end();
    }

    const int population_size = 100;
    std::vector<std::vector<bool>> population{converted_sol};

    // initial population based on given (greedy) solution
    for (int i = 1; i < population_size; ++i)
    {
        if (terminate_flag)
            break;
        population.push_back(mutate(converted_sol));
    }

    // precomputes if mutated solution is valid
    std::vector<bool> normal_pop(population_size);
    for (size_t i = 0; i < population.size() && !terminate_flag; ++i)
    {
        normal_pop[i] = deep_validate(graph, convert_to_normal(population[i]), k);
    }

    if (!terminate_flag)
    {
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);

        // sorts solutions based on fitness (the smaller solution the better)
        // invalid solutions are at the end
        std::sort(indices.begin(), indices.end(),
                  [&](size_t i, size_t j)
                  {
                      const auto &a = population[i];
                      const auto &b = population[j];
                      int count_a = std::count(a.begin(), a.end(), true);
                      int count_b = std::count(b.begin(), b.end(), true);
                      bool valid_a = normal_pop[i];
                      bool valid_b = normal_pop[j];
                      if (valid_a != valid_b)
                          return valid_a > valid_b;
                      return count_a < count_b;
                  });

        std::vector<std::vector<bool>> sorted_population(population.size());
        std::vector<bool> sorted_normal_pop(normal_pop.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            sorted_population[i] = population[indices[i]];
            sorted_normal_pop[i] = normal_pop[indices[i]];
        }
        population = std::move(sorted_population);
        normal_pop = std::move(sorted_normal_pop);
    }

    // optimizes till the end of times
    while (!terminate_flag)
    {
        population.resize(population_size / 2);
        int temp_size = population.size() - 1;

        while (!terminate_flag && population.size() < population_size)
        {
            std::uniform_int_distribution<> dis(0, temp_size);
            population.push_back(crossover(population[dis(gen)],
                                           population[dis(gen)]));
        }

        normal_pop.resize(population.size());
        for (size_t i = 0; i < population.size() && !terminate_flag; ++i)
        {
            normal_pop[i] = deep_validate(graph, convert_to_normal(population[i]), k);
        }

        if (!terminate_flag)
        {
            std::vector<size_t> indices(population.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::sort(indices.begin(), indices.end(),
                      [&](size_t i, size_t j)
                      {
                          const auto &a = population[i];
                          const auto &b = population[j];
                          int count_a = std::count(a.begin(), a.end(), true);
                          int count_b = std::count(b.begin(), b.end(), true);
                          bool valid_a = normal_pop[i];
                          bool valid_b = normal_pop[j];
                          if (valid_a != valid_b)
                              return valid_a > valid_b;
                          return count_a < count_b;
                      });

            std::vector<std::vector<bool>> sorted_population(population.size());
            std::vector<bool> sorted_normal_pop(normal_pop.size());
            for (size_t i = 0; i < indices.size(); ++i)
            {
                sorted_population[i] = population[indices[i]];
                sorted_normal_pop[i] = normal_pop[indices[i]];
            }
            population = std::move(sorted_population);
            normal_pop = std::move(sorted_normal_pop);
        }
    }

    return convert_to_normal(population[0]);
}

// when we try to run greedy algorithm on big k (for example k = 5)
// and the graph is big, then it is too slow
// that's why I use ramp up approach, so algorithms starts with k = min(2, k)
// and after solution is found then it tries to find solution for bigger k

std::vector<Node *> ramp_up_runner(std::vector<Node *> &graph, int k)
{
    int current_k = std::min(2, k);
    auto solution = greedy_solve(graph, current_k);

    while (!terminate_flag && current_k < k)
    {
        current_k++;
        for (Node *node : graph)
        {
            node->is_covered = false;
        }
        auto alternative = greedy_solve(graph, current_k);

        if (alternative.size() < solution.size())
        {
            solution = alternative;
        }
    }

    // after solution on desired k is found, we can run genetic_optimization
    if (!terminate_flag)
    {
        solution = genetic_optimization(graph, solution, k);
    }
    return solution;
}

// input read, graph generation and start of the program
void optilio_run()
{
    start_time = std::chrono::steady_clock::now().time_since_epoch().count() / 1e9;

    int n, m;
    std::cin >> n >> m;

    std::vector<Node *> graph;
    for (int i = 0; i < n; ++i)
    {
        graph.push_back(new Node(i));
    }

    for (int i = 0; i < m; ++i)
    {
        int u, v;
        std::cin >> u >> v;
        graph[u]->neighbors.push_back(graph[v]);
        graph[v]->neighbors.push_back(graph[u]);
    }

    std::string empty;
    std::getline(std::cin, empty);
    std::getline(std::cin, empty);

    int k;
    std::cin >> k;

    auto solution = ramp_up_runner(graph, k);

    std::cout << solution.size() << std::endl;
    for (Node *node : solution)
    {
        std::cout << node->id << " ";
    }
    std::cout << std::endl;

    for (Node *node : graph)
    {
        delete node;
    }
}

int main()
{
    optilio_run();
    return 0;
}