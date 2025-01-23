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
std::random_device rd;
std::mt19937 gen;

bool check_time()
{
    if ((std::chrono::steady_clock::now().time_since_epoch().count() / 1e9) - start_time >= 28)
    {
        terminate_flag = true;
        return true;
    }
    return false;
}

// get all nodes in range k, and also number of uncovered nodes in this range
std::pair<std::vector<Node *>, int> get_nodes_in_range(Node *start_node, int k)
{
    std::vector<Node *> output;
    std::queue<Node *> q;
    std::vector<int> distances;
    int uncovered = 0;
    std::unordered_set<int> visited;
    output.reserve(100);
    visited.reserve(1000);
    q.push(start_node);
    visited.insert(start_node->id);
    distances.push_back(0);
    int current_distance = 0;
    size_t nodes_at_current_level = 1;
    size_t nodes_at_next_level = 0;

    while (!q.empty() && current_distance <= k)
    {
        Node *node = q.front();
        q.pop();
        nodes_at_current_level--;

        output.push_back(node);

        if (!node->is_covered)
        {
            uncovered++;
        }

        if (current_distance < k)
        {
            Node **neighbor_end = node->neighbors.data() + node->neighbors.size();
            for (Node **neighbor_ptr = node->neighbors.data(); neighbor_ptr != neighbor_end; ++neighbor_ptr)
            {
                Node *neighbor = *neighbor_ptr;
                if (visited.insert(neighbor->id).second)
                {
                    q.push(neighbor);
                    nodes_at_next_level++;
                }
            }
        }

        if (nodes_at_current_level == 0)
        {
            current_distance++;
            nodes_at_current_level = nodes_at_next_level;
            nodes_at_next_level = 0;
        }
    }
    return {output, uncovered};
}

// marks all nodes in range k as covered
void cover_nodes(Node *start_node, int k)
{
    std::queue<Node *> q;
    std::unordered_set<int> visited;

    visited.reserve(1000);

    q.push(start_node);
    visited.insert(start_node->id);
    start_node->is_covered = true;

    int current_distance = 0;
    size_t nodes_at_current_level = 1;
    size_t nodes_at_next_level = 0;

    while (!q.empty() && current_distance <= k)
    {
        Node *node = q.front();
        q.pop();
        nodes_at_current_level--;

        if (current_distance < k)
        {
            Node **neighbor_end = node->neighbors.data() + node->neighbors.size();
            for (Node **neighbor_ptr = node->neighbors.data(); neighbor_ptr != neighbor_end; ++neighbor_ptr)
            {
                Node *neighbor = *neighbor_ptr;
                if (visited.insert(neighbor->id).second)
                {
                    neighbor->is_covered = true;
                    q.push(neighbor);
                    nodes_at_next_level++;
                }
            }
        }

        if (nodes_at_current_level == 0)
        {
            current_distance++;
            nodes_at_current_level = nodes_at_next_level;
            nodes_at_next_level = 0;
        }
    }
}

// faster cover if we give vector of nodes as input
void cover_vector(std::vector<Node *> &nodes)
{
    for (Node *node : nodes)
    {
        node->is_covered = true;
    }
}

// 1. automatically adds nodes without neighbors to the solution
// 2. then for each nodes with one neighbor (leaf node)
// calculates the best node from range k to add to solution

std::vector<Node *> reduction(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> initial_solution;

    for (Node *node : graph)
    {
        if (check_time())
        {
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
            auto [possible_nodes, dummy_cover_count] = get_nodes_in_range(node, k);

            // initially we add the only neighbor of leaf node
            Node *best_node = node->neighbors[0];
            auto [dummy_node, current_cover] = get_nodes_in_range(best_node, k);
            int highest_cover = current_cover;

            // my mind is gone, so let's say this is a medieval tournament
            for (Node *knight : possible_nodes)
            {
                if (check_time())
                {
                    return graph;
                }
                auto [_, cover_count] = get_nodes_in_range(knight, k);
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

// bad but quick solution
std::vector<Node *> ultra_greedy(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> solution;

    for (Node *node : graph)
    {
        if (node->is_covered)
            continue;

        solution.push_back(node);
        cover_nodes(node, k);
    }
    return solution;
}

// goes through the graph and adds nodes to the solution based on greedy approach
std::vector<Node *> greedy_pick(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> solution;

    for (Node *node : graph)
    {
        if (check_time())
        {
            return graph;
        }

        if (node->is_covered)
            continue;

        auto [possible_nodes, node_cover] = get_nodes_in_range(node, k);
        Node *best_node = node;
        int highest_cover = node_cover;

        for (Node *pos_node : possible_nodes)
        {
            if (check_time())
            {
                return graph;
            }
            auto [_, cover_count] = get_nodes_in_range(pos_node, k);
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

// validates solution
bool validate(std::vector<Node *> &graph, const std::vector<Node *> &solution, int k)
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

bool shallow_validate(std::vector<Node *> &graph)
{
    return std::all_of(graph.begin(), graph.end(), [](Node *node)
                       { return node->is_covered; });
}

void apply_solution(std::vector<Node *> &graph, const std::vector<Node *> &solution, int k)
{
    for (Node *node : graph)
    {
        node->is_covered;
    }
    for (Node *node : solution)
    {
        cover_nodes(node, k);
    }
}
std::vector<Node *> local_improvement(std::vector<Node *> &graph, const std::vector<Node *> &solution, int k)
{
    std::vector<Node *> alternative = solution;

    // choose random node
    std::uniform_int_distribution<> distribution(0, alternative.size() - 1);
    Node *node = alternative[distribution(gen)];

    // remove node from solution
    auto it = std::find(alternative.begin(), alternative.end(), node);
    if (it != alternative.end())
    {
        alternative.erase(it);
    }
    apply_solution(graph, alternative, k);

    // find new node which doesn't make solution invalid
    auto [possible_nodes, cover_count] = get_nodes_in_range(node, k);
    std::shuffle(possible_nodes.begin(), possible_nodes.end(), rd);
    Node *new_node = node;

    // pick randomly node which
    while (!possible_nodes.empty())
    {
        new_node = possible_nodes.back();
        possible_nodes.pop_back();
        cover_nodes(new_node, k);
        if (shallow_validate(graph))
        {
            alternative.push_back(new_node);
            return alternative;
        }
        apply_solution(graph, alternative, k);
    }
    return solution;
}

std::vector<Node *> deconstruction(std::vector<Node *> &graph, std::vector<Node *> &solution, int k, float probability = 0.2)
{
    if (check_time())
    {
        return solution;
    }
    std::vector<Node *> new_solution;
    std::uniform_real_distribution<> dist(0.0, 1.0);

    // reset graph
    for (Node *node : graph)
    {
        node->is_covered = false;
    }

    for (Node *node : solution)
    {
        if (dist(gen) > probability)
        {
            new_solution.push_back(node);
            cover_nodes(node, k);
        }
    }
    return new_solution;
}

// local search optimization
std::vector<Node *> local_search(std::vector<Node *> &graph, std::vector<Node *> &solution, int k)
{
    int counter = 0;
    while (!terminate_flag)
    {
        std::vector<Node *> alternative = local_improvement(graph, solution, k);
        if (check_time())
        {
            return solution;
        }
        // one in ten times destroy big chunk of graph
        if (counter % 10 == 0)
        {
            alternative = deconstruction(graph, alternative, k, 0.4);
        }
        else
        {
            alternative = deconstruction(graph, alternative, k, 0.2);
        }
        counter += 1;

        if (check_time())
        {
            return solution;
        }
        std::vector<Node *> greedy_fill = greedy_pick(graph, k);
        alternative.insert(alternative.end(), greedy_fill.begin(), greedy_fill.end());

        if (alternative.size() < solution.size())
        {
            solution = alternative;
        }
        check_time();
    }
    return solution;
}

// combines all parts into one function
std::vector<Node *> runner(std::vector<Node *> &graph, int k)
{
    std::vector<Node *> solution = ultra_greedy(graph, k);

    // graph reset
    for (Node *node : graph)
    {
        node->is_covered = false;
    }

    // reduction
    std::vector<Node *> reduced = reduction(graph, k);
    if (check_time())
    {
        return solution;
    }
    std::vector<Node *> alternative = greedy_pick(graph, k);
    alternative.insert(alternative.end(), reduced.begin(), reduced.end());

    if (alternative.size() < solution.size())
    {
        solution = alternative;
    }

    if (check_time())
    {
        return solution;
    }

    // graph reset
    for (Node *node : graph)
    {
        node->is_covered = false;
    }

    std::vector<Node *> optimized = local_search(graph, solution, k);
    if (optimized.size() < solution.size())
    {
        solution = optimized;
    }

    return solution;
}

// input read, graph generation and start of the program
void optilio_run()
{
    start_time = std::chrono::steady_clock::now().time_since_epoch().count() / 1e9;
    std::mt19937 gen = std::mt19937(rd());

    int n, m;
    std::cin >> n >> m;

    std::vector<Node *> graph(n);

    for (int i = 0; i < n; ++i)
    {
        graph[i] = new Node(i);
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

    auto solution = runner(graph, k);

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