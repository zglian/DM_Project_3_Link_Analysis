import os
from collections import defaultdict 
import numpy as np 
from easydict import EasyDict as edict
import networkx as nx 
import matplotlib.pyplot as plt
from typing import Tuple
from pprint import pprint

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

def PageRank(G:Graph, 
            max_iters:int, 
            damping_factor:float):

    N = G.N
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

def HITS(G:Graph, 
    max_iters:int)-> Tuple[np.array, np.array]:
    auths = np.ones(G.N)
    hubs = np.ones(G.N) 
    
    for _ in range(max_iters):
        new_auths = np.zeros_like(auths)
        new_hubs = np.zeros_like(hubs)
        for n in range(G.N):
            new_auths[n] = hubs[G.in_neighbors[n]].sum()
            new_hubs[n] = auths[G.out_neighbors[n]].sum()
        # if norm == 'L1':
        auths = new_auths / np.sum(new_auths)
        hubs = new_hubs / np.sum(new_hubs)
    return auths, hubs 

def SimRank(G: Graph, 
            max_iters:int, 
            decay_factor:float):
    # SimRank_sum = the sum of SimRank value of all in-neighbor pairs (SimRank value is from the previous iteration)
    C = decay_factor 
    def update_simrank(a:int, b:int, simRank: np.array):
        if a == b: 
            return 1    
        a_in_neighbors = G.in_neighbors[a] 
        b_in_neighbors = G.in_neighbors[b] 
        a_in_size, b_in_size = len(a_in_neighbors), len(b_in_neighbors)
        if not a_in_size or not b_in_size: #if no in_neightbors
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
                newSimRank[a, b] = newSimRank[b, a] = update_simrank(a, b, simRank)
        simRank = newSimRank.copy() 
    return simRank    


#input
filedir ='./dataset/'
edges_data = {} 

for filename in os.listdir(filedir):
    edges = [] 
    filepath = os.path.join(filedir, filename) 
    print(f'reading {filepath}...')

    filepreff= filename.split('.')[0]
    if filename.startswith('graph') or filename.startswith('rev_graph'):
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

# D
damping_factor = 0.1
# C
decay_factor = 0.7
iteration = 30


Graphs = {}
revGraphs = {}
for fname, edges in edges_data:
    G = Graph(edges)
    Graphs[fname] = G
    if fname.startswith('rev'):
        revGraphs[fname] = G 
    print(fname, f': graph with {G.N} nodes and {len(G.edges)} edges')


for filename, g in Graphs.items():
    print("=====" + filename + "=====")
    pagerank = PageRank(g, iteration, damping_factor)
    pagerank = np.array2string(pagerank, precision = 3)
    print("Pagerank:\n" + pagerank)

    authority, hub = HITS(g, iteration)
    authority = np.array2string(authority, precision = 3)
    hub = np.array2string(hub, precision = 3)
    print("\nAuthority:\n" + authority)
    print("\nHub:\n" + hub)

    if filename != "graph_6" and filename != "ibm-5000":
        simrank = SimRank(g, iteration, damping_factor)
        simrank = np.array2string(simrank, precision = 3)
        print("\nSimrank:\n" + simrank)

    
    
