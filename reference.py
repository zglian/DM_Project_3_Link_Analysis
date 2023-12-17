import os
from collections import defaultdict 
import numpy as np 
from easydict import EasyDict as edict

filedir ='./dataset/'
edges_data = {} 

for filename in os.listdir(filedir):
    edges = [] 
    filepath = os.path.join(filedir, filename) 
    print(f'reading {filepath}...')

    filepreff= filename.split('.')[0]
    if filename.startswith('graph'):
        with open(filepath, 'r') as f: 
            for line in f.readlines():
                line = line.strip()
                edge = line.split(',')
                edges.append(edge)
        edges_data[filepreff] = edges
                
    elif filename.startswith('ibm'):
        with open(filepath, 'r') as f: 
            for line in f.readlines():
                line = line.strip()
                edge = line.split()[1:]
                edges.append(edge)
        edges_data[filepreff] = edges
edges_data = sorted(edges_data.items(), key = lambda x:x[0])
class Graph:
    def __init__(self, edges):
        self.out_neighbors = defaultdict(list)
        self.in_neighbors = defaultdict(list)
        self.edges = edges 
        nodes = set()
        for u, v in edges:
            nodes.add(u); nodes.add(v) 
        nodes = sorted(nodes)
        # print(nodes[:10])    
        nodesmap = {node:nodeidx for nodeidx, node in enumerate(nodes)}
        for u, v in edges:
            u, v = nodesmap[u], nodesmap[v]
            self.out_neighbors[u].append(v)
            self.in_neighbors[v].append(u)  
        self.N = len(nodes)


# damping factor 
D = 0.15 
# decay factor 
C = 0.9
# number of iterations
T = 100 


from pprint import pprint
Graphs = {}
revGraphs = {}
for fname, edges in edges_data:
    G = Graph(edges)
    Graphs[fname] = G
    if fname.startswith('rev'):
        revGraphs[fname] = G 
    print(fname, f': graph with {G.N} nodes and {len(G.edges)} edges')

def nodeid(node:str):
    return int(node)-1 
@timer 
def PageRank(G:Graph, 
            max_iters:int, 
            damping_factor:float):
    """
    Args:
        G (networkx.classes.graph.Graph): 
        max_iters (int): number of iters 
        damping_factor (float): 
        The PageRank theory holds that an imaginary surfer who is randomly clicking on links will eventually stop clicking. The probability, at any step, that the person will continue is a damping factor d.
    """
    N = G.N
    # if N == 0:
    #     raise ValueError('Empty Graph')
    PageRanksHistory = []
    d = damping_factor
    PageRanks = np.full(N, 1/N)   #將所有節點的權重初始化為1/N
    for iter in range(max_iters):
        newPageRanks = np.zeros(N)
        for i in range(N):
            for n in G.in_neighbors[i]:
                newPageRanks[i] += PageRanks[n] / len(G.out_neighbors[n])
        PageRanks =  d/N + (1-d) * newPageRanks
    PageRanks = PageRanks / (PageRanks.sum())
    return PageRanks
from typing import Tuple
@timer
def HITS(G:Graph, 
    max_iters:int, 
    norm = 'L1')-> Tuple[np.array, np.array]:
    """
    HITS(Hyperlink-induced topic search)
    Authority: Providing valuable infor on certain topic 
    Hub: Give good supports to those pages with high authority
    - A good hub increases the authority weight of the pages it points. 
    - A good authority increases the hub weight of the pages that point to it. 
    The idea is then to apply the two operations above alternatively until equilibrium values for the hub and authority weights are reached.
    Args:
        G (Graph): the given subgraph 
    Returns:
        Tuple(np.array, np.array): Auth, Hub Vectors 
            Auth: shape (N, ) Auth[n] is the authority score of node n
            Hub: shape (N, )  Similarly, Hub[n] is the hub score of node n
    """
    auths = np.ones(G.N)
    hubs = np.ones(G.N) 
    def get_update_Auth(n):
        # authority: the node being pointed to 越多人指向他越高分
        return hubs[G.in_neighbors[n]].sum()
    def get_update_Hub(n):
        return auths[G.out_neighbors[n]].sum()
    
    for _ in range(max_iters):
        new_auths = np.zeros_like(auths)
        new_hubs = np.zeros_like(hubs)
        for n in range(G.N):
            new_auths[n] = get_update_Auth(n)
            new_hubs[n] = get_update_Hub(n)
        if norm == 'L1':
            auths = new_auths / np.sum(new_auths)
            hubs = new_hubs / np.sum(new_hubs)
        else: # root of sum of squares
            # wiki: L2 norm 
            # https://en.wikipedia.org/wiki/HITS_algorithm
            auths = new_auths / np.sqrt(np.sum(new_auths**2))
            hubs = new_hubs / np.sqrt(np.sum(new_hubs**2))
    
    return auths, hubs 
@timer 
def SimRank(G: Graph, 
            max_iters:int, 
            decay_factor:float):
    # SimRank_sum = the sum of SimRank value of all in-neighbor pairs (SimRank value is from the previous iteration)
    C = decay_factor 
    def get_update_simrank(
                    a:int, 
                    b:int, 
                    simRank: np.array):
        if a == b: 
            return 1    
        a_in_neighbors = G.in_neighbors[a] # I_i(a)
        b_in_neighbors = G.in_neighbors[b] # I_j(b)
        a_in_size, b_in_size = len(a_in_neighbors), len(b_in_neighbors)
        if not a_in_size or not b_in_size:
            return 0
        temp = 0 
        for i in a_in_neighbors:
            for j in b_in_neighbors:
                temp += simRank[i, j]
        # scaling the simRank 
        return C * temp / (a_in_size * b_in_size) 
                        
    simRank = np.zeros((G.N, G.N))
    for iter in range(max_iters):
        newSimRank = np.zeros_like(simRank)
        for a in range(G.N):
            for b in range(a, G.N):
                newSimRank[a, b] = newSimRank[b, a] = get_update_simrank(a, b, simRank)
        simRank = newSimRank.copy() 
    return simRank    
for filename, g in Graphs.items():
    print('============')
    print(filename)
    # T, D, C = 100, 0.15, 0.9
    # pagerank = PageRank(g, max_iters=T, damping_factor = D)
    # auths, hubs = HITS(g, max_iters=T)
    # if g.N < 1000:
    #     simrank= SimRank(g, max_iters=T, decay_factor = C)
    
    
    print('------------')
    T, D, C = 40, 0.3, 1
    pagerank = PageRank(g, max_iters=T, damping_factor = D)
    auths, hubs = HITS(g, max_iters=T)
   
    np.set_printoptions(precision=3, 
                        threshold=np.inf)
    pagerank = np.array2string(pagerank, precision=3)
    auth = np.array2string(auths, precision=3)
    hubs= np.array2string(hubs, precision=3)
    
    
    # if g.N < 10:
    simrank= SimRank(g, max_iters=T, decay_factor = C)
    simrank = np.array2string(simrank, precision=3)
    print('pagerank:\n', pagerank)
    print('auth:\n', auth)
    print('hub:\n', hubs)
    print('simrank:\n', simrank)

# Which year's hyperparams to use 
# hyperparams 

Hyperparams ={2021: {'D': 0.15, 'C': 0.9, 'T': 100}, 
              2022: {'D':0.1, 'C': 0.7, 'T': 30}}
filedir ='./data/'

def save_and_display(year_hyper):
    result_root = f'./results/{year_hyper}'
    os.makedirs(result_root, exist_ok=True)
    prec = 3 
    D = Hyperparams[year_hyper]['D']
    C = Hyperparams[year_hyper]['C']
    T = Hyperparams[year_hyper]['T']
    
    np.set_printoptions(precision=prec, 
                        threshold=np.inf)
    Graphs.update(revGraphs)
    for gname, g in Graphs.items():
        file_prefix = gname.split('.')[0]
        gpath = os.path.join(result_root, file_prefix)
        os.makedirs(gpath, exist_ok=True)

        print(f'==== Graph: {gname} ===')
        filename = f'{gname}_PageRank.txt'
        pagerank = PageRank(g, max_iters = T, damping_factor = D)
        np.savetxt(os.path.join(gpath, filename), pagerank, fmt = f'%.{prec}f', newline=' ')
        print('🪬 pagerank:\n', pagerank)

        filename = f'{gname}_HITS'
        auths, hubs = HITS(g, max_iters = T)
        np.savetxt(os.path.join(gpath, filename+'_authority.txt'), auths, fmt=f'%.{prec}f',newline=' ' )
        np.savetxt(os.path.join(gpath, filename+'_hub.txt'), hubs, fmt=f'%.{prec}f', newline=' ')
        print('🪬 auths:\n', auths)
        print('🪬 hubs:\n', hubs)

        # avoid running SimRank on large graphs 
        if gname.startswith('ibm') or gname.startswith('graph_6'):
            continue 

        filename = f'{gname}_SimRank.txt'
        simrank = SimRank(g, max_iters = T, decay_factor = C)
        # matrix 
        np.savetxt(os.path.join(gpath, filename), simrank, fmt=f'%.{prec}f')
        print('🪬 simrank:\n', simrank)
save_and_display(year_hyper = 2022)



import networkx as nx 
import matplotlib.pyplot as plt

def save_graph(raw_g: Graph, gname:str):
    G = nx.DiGraph()        
    for edge in raw_g.edges:
        # print(edge)
        G.add_edge(*edge)
    nx.draw(G, with_labels=True)
    # plt.show(block=False)
    plt.savefig(f"{gname}.png", format="PNG")
    plt.clf() # clean plots

    
for idx, (gname, g) in enumerate(Graphs.items()):
    g_prefix = gname.split('.')[0]
    save_graph(raw_g = g, gname = gname)
    # if int(g_prefix.split('_')[1]) == 5: 
    #     continue 
    # if g_prefix.split('_')[0] == "ibm":
    #     continue
for idx, (gname, g) in enumerate(revGraphs.items()):
    g_prefix = gname.split('.')[0]
    save_graph(raw_g = g, gname = gname)