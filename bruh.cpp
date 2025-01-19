#include <iostream>
#include <vector>
#include <set>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <random>
#include <memory>

class Node
{
public:
    int id;
    std::set<std::shared_ptr<Node>> nearest_neighbors;
    std::set<std::shared_ptr<Node>> nodes_in_range;
    bool is_covered;
    int cover_count;

    explicit Node(int id) : id(id), is_covered(false), cover_count(0) {}

    std::set<std::shared_ptr<Node>> get_nearest()
    {
        std::set<std::shared_ptr<Node>> result;
        for (const auto &node : nearest_neighbors)
        {
            if (!node->is_covered)
            {
                result.insert(node);
            }
        }
        return result;
    }

    std::set<std::shared_ptr<Node>> get_nodes_in_range()
    {
        std::set<std::shared_ptr<Node>> result;
        for (const auto &node : nodes_in_range)
        {
            if (!node->is_covered)
            {
                result.insert(node);
            }
        }
        return result;
    }
};

std::vector<std::shared_ptr<Node>> graph_from_input()
{
    int n, m;
    std::cin >> n >> m;

    std::vector<std::shared_ptr<Node>> graph;
    for (int i = 0; i < n; i++)
    {
        graph.push_back(std::make_shared<Node>(i));
    }

    for (int i = 0; i < m; i++)
    {
        int u, v;
        std::cin >> u >> v;
        graph[u]->nearest_neighbors.insert(graph[v]);
        graph[v]->nearest_neighbors.insert(graph[u]);
    }

    std::string empty;
    std::getline(std::cin, empty);
    std::getline(std::cin, empty);

    return graph;
}

std::vector<std::shared_ptr<Node>> clone_graph(const std::vector<std::shared_ptr<Node>> &nodes)
{
    std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> node_mapping;

    std::vector<std::shared_ptr<Node>> result;
    for (const auto &original_node : nodes)
    {
        auto new_node = std::make_shared<Node>(original_node->id);
        new_node->is_covered = original_node->is_covered;
        new_node->cover_count = original_node->cover_count;
        node_mapping[original_node] = new_node;
        result.push_back(new_node);
    }

    for (size_t i = 0; i < nodes.size(); i++)
    {
        const auto &original_node = nodes[i];
        auto new_node = result[i];

        for (const auto &neighbor : original_node->nearest_neighbors)
        {
            if (node_mapping.find(neighbor) != node_mapping.end())
            {
                new_node->nearest_neighbors.insert(node_mapping[neighbor]);
            }
        }

        for (const auto &node : original_node->nodes_in_range)
        {
            if (node_mapping.find(node) != node_mapping.end())
            {
                new_node->nodes_in_range.insert(node_mapping[node]);
            }
        }
    }

    return result;
}

std::set<std::shared_ptr<Node>> get_nodes_in_range(std::shared_ptr<Node> start_node, int k)
{
    std::set<std::shared_ptr<Node>> nodes_within_range;
    std::queue<std::pair<std::shared_ptr<Node>, int>> queue;
    std::set<std::shared_ptr<Node>> visited;

    queue.push({start_node, 0});
    visited.insert(start_node);

    while (!queue.empty())
    {
        auto current = queue.front();
        queue.pop();
        auto current_node = current.first;
        int depth = current.second;

        if (depth > k)
            continue;

        nodes_within_range.insert(current_node);
        for (const auto &neighbor : current_node->nearest_neighbors)
        {
            if (visited.find(neighbor) == visited.end())
            {
                visited.insert(neighbor);
                queue.push({neighbor, depth + 1});
            }
        }
    }

    return nodes_within_range;
}

void extend_graph(std::vector<std::shared_ptr<Node>> &graph, int k)
{
    size_t n = graph.size();
    std::vector<std::vector<int>> distances(n, std::vector<int>(n, INT_MAX));

    for (size_t start = 0; start < n; start++)
    {
        std::queue<std::pair<int, int>> q;
        q.push(std::make_pair(start, 0));
        distances[start][start] = 0;

        while (!q.empty())
        {
            int current = q.front().first;
            int dist = q.front().second;
            q.pop();

            if (dist == k)
                continue;

            for (const auto &neighbor : graph[current]->nearest_neighbors)
            {
                int neighbor_id = neighbor->id;
                if (distances[start][neighbor_id] == INT_MAX)
                {
                    distances[start][neighbor_id] = dist + 1;
                    q.push(std::make_pair(neighbor_id, dist + 1));
                }
            }
        }
    }

    for (size_t i = 0; i < n; i++)
    {
        graph[i]->nodes_in_range.clear();
        for (size_t j = 0; j < n; j++)
        {
            if (distances[i][j] <= k)
            {
                graph[i]->nodes_in_range.insert(graph[j]);
            }
        }
        graph[i]->cover_count = graph[i]->nodes_in_range.size();
    }
}

void reduce_graph(std::vector<std::shared_ptr<Node>> &graph, std::shared_ptr<Node> del_node)
{
    auto nodes_in_range = del_node->get_nodes_in_range();
    for (const auto &neighbor : nodes_in_range)
    {
        neighbor->is_covered = true;
    }

    graph.erase(
        std::remove_if(graph.begin(), graph.end(),
                       [&](const std::shared_ptr<Node> &node)
                       {
                           return nodes_in_range.find(node) != nodes_in_range.end();
                       }),
        graph.end());
}

struct CompareNodes
{
    bool operator()(const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b)
    {
        return a->get_nodes_in_range().size() < b->get_nodes_in_range().size();
    }
};

std::set<std::shared_ptr<Node>> greedy_pick(std::vector<std::shared_ptr<Node>> &graph)
{
    std::set<std::shared_ptr<Node>> solution;
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, CompareNodes> pq;

    for (const auto &node : graph)
    {
        pq.push(node);
    }

    while (!graph.empty() && !pq.empty())
    {
        auto best_node = pq.top();
        pq.pop();

        if (std::find(graph.begin(), graph.end(), best_node) == graph.end())
        {
            continue;
        }

        solution.insert(best_node);
        reduce_graph(graph, best_node);
    }

    return solution;
}

std::set<std::shared_ptr<Node>> leaf_reduction(std::vector<std::shared_ptr<Node>> &graph)
{
    std::set<std::shared_ptr<Node>> init_sol;
    size_t i = 0;

    while (i < graph.size())
    {
        auto node = graph[i];
        auto nearest = node->get_nearest();

        if (nearest.empty())
        {
            init_sol.insert(node);
            i++;
        }
        else if (nearest.size() == 1)
        {
            auto best = *std::max_element(
                node->nodes_in_range.begin(),
                node->nodes_in_range.end(),
                [](const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b)
                {
                    return a->cover_count < b->cover_count;
                });
            init_sol.insert(best);
            reduce_graph(graph, best);
        }
        else
        {
            i++;
        }
    }

    return init_sol;
}

std::pair<std::vector<std::shared_ptr<Node>>, std::set<std::shared_ptr<Node>>>
greedy_deconstruction(const std::vector<std::shared_ptr<Node>> &graph,
                      const std::set<std::shared_ptr<Node>> &nodes,
                      double percentage = 0.25)
{
    std::vector<std::shared_ptr<Node>> temp_nodes(nodes.begin(), nodes.end());
    std::sort(temp_nodes.begin(), temp_nodes.end(),
              [](const std::shared_ptr<Node> &a, const std::shared_ptr<Node> &b)
              {
                  return a->cover_count < b->cover_count;
              });

    size_t first_idx = static_cast<size_t>(temp_nodes.size() * percentage);
    temp_nodes.erase(temp_nodes.begin(), temp_nodes.begin() + first_idx);

    auto temp_graph = clone_graph(graph);
    std::set<std::shared_ptr<Node>> result_nodes;

    for (const auto &node : temp_nodes)
    {
        auto corresponding_node = temp_graph[node->id];
        if (std::find(temp_graph.begin(), temp_graph.end(), corresponding_node) != temp_graph.end())
        {
            result_nodes.insert(corresponding_node);
            reduce_graph(temp_graph, corresponding_node);
        }
    }

    return {temp_graph, result_nodes};
}

std::pair<std::vector<std::shared_ptr<Node>>, std::set<std::shared_ptr<Node>>>
random_deconstruction(const std::vector<std::shared_ptr<Node>> &graph,
                      const std::set<std::shared_ptr<Node>> &nodes,
                      double percentage = 0.2)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, 1);

    auto temp_graph = clone_graph(graph);
    std::set<std::shared_ptr<Node>> temp_nodes;

    for (const auto &node : nodes)
    {
        if (dis(gen) < percentage)
        {
            auto corresponding_node = temp_graph[node->id];
            temp_nodes.insert(corresponding_node);
        }
    }

    for (const auto &node : temp_nodes)
    {
        if (std::find(temp_graph.begin(), temp_graph.end(), node) != temp_graph.end())
        {
            reduce_graph(temp_graph, node);
        }
    }

    return {temp_graph, temp_nodes};
}

std::set<std::shared_ptr<Node>> algorithm(std::vector<std::shared_ptr<Node>> &original_graph, int k)
{
    auto start_time = std::chrono::steady_clock::now();
    extend_graph(original_graph, k);
    auto graph = clone_graph(original_graph);

    std::set<std::shared_ptr<Node>> best_solution = leaf_reduction(graph);
    auto greedy_solution = greedy_pick(graph);
    best_solution.insert(greedy_solution.begin(), greedy_solution.end());

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start_time)
               .count() < 20)
    {
        std::pair<std::vector<std::shared_ptr<Node>>, std::set<std::shared_ptr<Node>>> result1 =
            greedy_deconstruction(original_graph, best_solution);
        auto greedy_nodes1 = greedy_pick(result1.first);
        result1.second.insert(greedy_nodes1.begin(), greedy_nodes1.end());
        if (result1.second.size() < best_solution.size())
        {
            best_solution = result1.second;
        }

        std::pair<std::vector<std::shared_ptr<Node>>, std::set<std::shared_ptr<Node>>> result2 =
            random_deconstruction(original_graph, best_solution);
        auto greedy_nodes2 = greedy_pick(result2.first);
        result2.second.insert(greedy_nodes2.begin(), greedy_nodes2.end());
        if (result2.second.size() < best_solution.size())
        {
            best_solution = result2.second;
        }
    }

    return best_solution;
}

int main()
{
    auto graph = graph_from_input();
    int k;
    std::cin >> k;

    auto solution = algorithm(graph, k);

    std::cout << solution.size() << std::endl;
    for (const auto &node : solution)
    {
        std::cout << node->id << " ";
    }
    std::cout << std::endl;

    return 0;
}