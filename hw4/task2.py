from pyspark import SparkContext, SparkConf
from collections import OrderedDict
import sys
from itertools import permutations
from copy import deepcopy

# input: graph,start point
# output:
# nodes_list: with nodes in the same level in one list
# parent: each nodes' parent list

def BFS(graph,start):
    seen = {start} | set(graph[start])
    mul_level_node = graph[start]
    level_list = []
    parent = {node:[start] for node in mul_level_node}
    short_path = {node:1 for node in seen}
    while len(mul_level_node)>0:
        level_list.append(mul_level_node)
        seen = seen|set(mul_level_node)
        node_iter = mul_level_node
        mul_level_node = []
        for node in node_iter:
            sons = graph[node] #check each node's son nodes
            for son in sons:
                if son not in seen: #if the son node haven't been seen, add to the next level node list
                    try:
                        parent[son].append(node)
                    except:
                        parent[son] = [node]
                    try:
                        short_path[son] += short_path[node]
                    except:
                        short_path[son] = short_path[node]
                    if son not in mul_level_node:
                        mul_level_node.append(son)
    return level_list,parent,short_path

def BFS_retrave(graph,s):
    queue = []
    queue.append(s)
    seen = set()
    while (len(queue) > 0):
        vertex = queue.pop(0)
        nodes = graph[vertex]
        for w in nodes:
            if w not in seen:
                queue.append(w)
                seen.add(w)
    return seen

# we want to generate the element of list reversely
def verse_generate(mul_list):
    num = len(mul_list)
    for i in range(1,num+1):
        yield mul_list[i*(-1)]

# for each starting point, generate the weight((small_id,large_id):weight)
def weight(graph,start):
    all_nodes = graph.keys()
    credit = {raw_node:1 for raw_node in all_nodes if raw_node!=start}
    weight = {}
    level_list,parents,short_path = BFS(graph,start)
    for node_list in verse_generate(level_list):
        for node in node_list:
            for parent in parents[node]:
                if node < parent:
                    weight[(node,parent)] = credit[node]*(short_path[parent]/short_path[node])
                elif node > parent:
                    weight[(parent,node)] = credit[node]*(short_path[parent]/short_path[node])
                if parent != start:
                    credit[parent] += credit[node]*(short_path[parent]/short_path[node])
    return weight

# input: graph --- {node1:[node2..]...}
# output: edges with the largest betweenes {(node1,node2):betweeness...}
def get_betweeness(graph):
    betweenness = {}
    if len(graph.keys()) == 1: # if this is an isolated community
        return {}
    else:
        for node in graph.keys():
            edge_weight = weight(graph, node)
            for edge in edge_weight:
                try:
                    betweenness[edge] += edge_weight[edge]
                except:
                    betweenness[edge] = edge_weight[edge]

        betweenness_half = {key: betweenness[key] / 2 for key in betweenness.keys()}
        max_betweenes = max(betweenness_half.values())
        large_edge = {edges:max_betweenes for edges in betweenness_half if betweenness_half[edges] == max_betweenes}
        return large_edge

# input:graph need to cut;large_edge dictionary
# output:1){nodes_tuple:graph};2){nodes:each graph's largest betweennes and their edge}
# {nodes_tuple:{edge:max_betweeness}..}
def graph_cutting(graph,edges):
    new_graph = deepcopy(graph)
    nodes = set(node for edges in edges.keys() for node in edges ) #get all of nodes need to cut
    new_comunities = {}
    large_edge = {}
    for edge in edges.keys(): #delete those edges
        new_graph[edge[0]].remove(edge[1])
        new_graph[edge[1]].remove(edge[0])
    for point in nodes:
        all_nodes = BFS_retrave(new_graph,point)
        if set(all_nodes) == graph.keys():
            new_comunities[tuple(graph.keys())] = new_graph
            large_edge[tuple(graph.keys())] = get_betweeness(new_graph)
            break
        else:
            sub_nodes = set(all_nodes)
            if tuple(sub_nodes) not in new_comunities.keys():
                sub_graph = {node: new_graph[node] for node in sub_nodes}
                new_comunities[tuple(sub_nodes)] = sub_graph
                large_edge[tuple(sub_nodes)] = get_betweeness(sub_graph)
    return new_comunities,large_edge



# input: original_graph, all of current communities
# output: sum of modularity
def sum_modularity(original_graph,communities):
    m_2 = sum(len(nodes) for nodes in original_graph.values())
    Q = 0
    for community in communities:
        if len(community) == 1:
            continue
        else:
            pairs = permutations(community,2)
            for pair in pairs:
                if pair[1] in original_graph[pair[0]]:
                    A = 1
                else:
                    A = 0
                Q += A - (len(original_graph[pair[0]]) * len(original_graph[pair[1]]))/m_2
    return Q*(1/m_2)




# input_file = 'ub_sample_data.csv'
# betweenness_output = 'test2_1.txt'
# community_output = 'test2_2.txt'
# threshold = 7
if __name__ == '__main__':
    threshold = sys.argv[1]
    input_file = sys.argv[2]
    betweenness_output = sys.argv[3]
    community_output = sys.argv[4]

    conf = SparkConf().setMaster("local") \
            .setAppName("hw4_task1") \
            .set("spark.executor.memory", "15g") \
            .set('spark.executor.cores', '10') \
            .set("spark.driver.memory", "20g") \
            .set('spark.executor.instances', '20')

    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    rdd = sc.textFile(input_file)\
        .map(lambda x:x.split(',')).cache()

    header = rdd.first()

    # get user rdd: (user_id,set(business_id1...))
    user_rdd = rdd.filter(lambda x: x!= header)\
        .map(lambda x:(x[0],x[1]))\
        .groupByKey()\
        .mapValues(lambda x: set(x))


    # get the user_id edge
    # ((user1,business_list1),(user2,business_list2)) --> ((user1,user2),len) --> filter --> (user1,user2)
    # threshold = 7
    edge_rdd_du = user_rdd.cartesian(user_rdd)\
        .map(lambda x: ((x[0][0],x[1][0]),len(x[0][1]&x[1][1])))\
        .filter(lambda x: x[1]>=threshold)\
        .filter(lambda x: x[0][0] != x[0][1])\
        .map(lambda x:(x[0][0],x[0][1])).distinct()

    # {user1:[user2...]},gives the nodes and its neighbor nodes
    node_dict = dict(edge_rdd_du.groupByKey().mapValues(lambda x:list(x)).collectAsMap())

    # get all of node
    node_set = set(node_dict.keys())

    # calculate the betweenness
    betweenness = {}
    for node in node_set:
        edge_weight = weight(node_dict,node)
        for edge in edge_weight:
            try:
                betweenness[edge] += edge_weight[edge]
            except:
                betweenness[edge] = edge_weight[edge]

    betweenness_half = {key:betweenness[key]/2 for key in betweenness.keys()}

    output = OrderedDict(sorted(betweenness_half.items(),key=lambda x:(-x[1],x[0][0])))

    with open(betweenness_output,'w+') as f:
        for key,value in output.items():
            f.write(str(key))
            f.write(',')
            f.write(str(round(value,5)))
            f.write('\n')

    # optimize the modularity

    opt_community = []
    node_graph_dict = {}
    finish = []
    for node in node_set:
        if node not in finish:
            node_community = BFS_retrave(node_dict,node)
            node_graph_dict[tuple(node_community)] = {key:node_dict[key] for key in node_community}
            opt_community.append(tuple(node_community))
            finish += node_community


    # opt_community = [tuple(node_set)]
    # graph_to_cut = node_dict  # initialize the graph should be cut
    graph_betweeness = {tuple(node_set): get_betweeness(node_dict) for node_set,node_dict in node_graph_dict.items()}
    # node_graph_dict = {tuple(node_set): node_dict}
    # next_nodes = tuple(node_set)
    Q_0 = sum_modularity(node_dict, opt_community)
    Q_1 = Q_0
    while True:
        largest = 0
        for nodes, edges in graph_betweeness.items():
            if edges:
                value = max(edges.values())
                if value > largest:
                    largest = value
                    next_nodes = nodes
        graph_to_cut = node_graph_dict[next_nodes]
        Q_0 = Q_1
        small_graph, large_betweeness = graph_cutting(graph_to_cut, graph_betweeness[next_nodes])
        next_community = deepcopy(opt_community)
        next_community.remove(next_nodes)
        next_community += list(small_graph.keys())
        Q_1 = sum_modularity(node_dict, next_community)
        if Q_1 - Q_0 < - 0.0000001:
            break
        else:
            opt_community.remove(next_nodes)
            opt_community += list(small_graph.keys())
            graph_betweeness.pop(next_nodes)
            for key, value in large_betweeness.items():  # update the betweenes dictionary
                graph_betweeness[key] = value
            node_graph_dict.pop(next_nodes)
            for key, value in small_graph.items():  # update the sub_graph dictionary
                node_graph_dict[key] = value

    communities = sorted([sorted(k) for k in opt_community],key=lambda x: (len(x),x[0]))
    with open(community_output,'w+') as f:
        for community in communities:
            community_ls = ["'"+word+"'" for word in community]
            f.write(', '.join(community_ls))
            f.write('\n')




