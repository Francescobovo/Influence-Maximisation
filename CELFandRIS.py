# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:15:35 2018

@author: Francesco Bovo
"""
import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import pandas as pd
import time
import random
from collections import Counter
import igraph.test

# testing results
G = pd.read_csv(r'C:\Users\Francesco\OneDrive\Documents\Leiden\Courses\Social Network Analysis\project\paper\out.moreno_oz_oz', delimiter = " ", index_col = False)
celf_output = celf(G = G, k = 20, p = 0.2, mc = 10000, weight = False)

ris_output = ris(G = G, k = 20, p = 0.2, mc = 10000, weight = False)

celf_spread = IC(G = G, S = celf_output[0], p = 0.2, mc = 10000, weight = False)
ris_spread = IC(G = G, S = ris_output[0], p = 0.2, mc = 10000, weight = False)


ris_spread = [0] * 10
for i in range(10):
    ris_output = ris(G = G, k = 20, p = 0.2, mc = 10000)
    ris_spread[i] = IC(G = G, S = ris_output[0], p = 0.2, mc = 10000)

mean(ris_spread)
spread_list = [0] * 4
spread_list[0] = 'celf'
spread_list[2] = 'ris'
spread_list[1] = celf_spread
spread_list[3] = ris_spread
spread_list
results = [0] * 5
results[0] = celf_output[0]
results[1] = celf_output[1]
results[2] = ris_output[0]
results[3] = ris_output[1]
results[4] = spread_list
pd.DataFrame(results).to_csv("results_ozuw",index=False)

#########


randweights = [0] * len(G)
for i in range(len(G)):
    randweights[i] = random.randint(1,5)
G['weight'] = randweights

# testing costs
costdataset = [0] * len(np.unique(G))
for i in range(len(np.unique(G))):
   costdataset[i] = random.randint(1,10)
   

               
def IC(G,S,p=0.5,mc=1000, weight = False):
    """
    Input:  G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            S:  Set of seed nodes
            p:  Disease propagation probability
            mc: Number of Monte-Carlo simulations
    Output: Average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for _ in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:
            
            # Get edges that flow out of each newly active node
            temp = G.loc[G['source'].isin(new_active)]

            # Extract the out-neighbors of those nodes
            targets = temp['target'].tolist()
            
            # Weights or no Weights?
            if weight == True:
                # Extract the weights of the out-neighbors of those nodes
                weights = temp['weight'].tolist()
            
                # Weight value indicates how many times it is possible for the node to activate the out-neighbour
                # probability of not activating a node w times, (1-p)^w
                # therefore probability of activiating a node at least once out of w times = 1 - (1-p)^w, or simply flip the < to >
                success = [0] * len(targets)
                for i in range(len(targets)):
                    success[i] = np.random.uniform(0,1) > ((1-p)**weights[i])
            elif weight == False:
                success  = np.random.uniform(0,1,len(targets)) < p
                
            # Determine those neighbors that become infected
            new_ones = np.extract(success, targets)
            
            # Create a list of nodes that weren't previously activated
            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return(np.mean(spread))    
    
def celf(G,k,cost=[1] * len(np.unique(G['source'])),p=0.5,mc=1000, weight = False):   
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of Monte-Carlo simulations
    Return: A seed set of nodes as an approximate solution to the IM problem
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Compute marginal gain for each node       
    candidates, start_time = np.unique(G['source']), time.time()
    marg_gain = [IC(G,[candidates[i]],p=p,mc=mc,weight=weight)/cost[i] for i in range(len(candidates))]
    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(candidates,marg_gain), key = lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    timelapse = [time.time()-start_time]
    
    # --------------------
    # Find the next k-1 nodes using the CELF list-sorting procedure
    # --------------------
    
    for _ in range(k-1):    

        check = False
        
        while not check:
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC(G,S+[current],p=p,mc=mc,weight=False) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = Q[0][0] == current

        # Select the next node
        S.append(Q[0][0])
        spread = Q[0][1]
        timelapse.append(time.time() - start_time)
        
        # Remove the selected node from the list
        Q = Q[1:]
    
    return(sorted(S),timelapse)
    
    
def get_RRS(G,p, weight = False):   
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: A random reverse reachable set expressed as a list of nodes
    """
    
    # Step 1. Select random source node
    source = random.choice(np.unique(G['source']))
    
    # Step 2. Get an instance of g from G by sampling edges
    if weight == True:
        # Extract the weights 
        weights = G['weight'].tolist()
            
        # Weight value indicates how many times it is possible for the node to activate the out-neighbour
        # probabilty of not activating a node w times, (1-p)^w
        # therefore probability of activating a node at least once out of w times = 1 - (1-p)^w, or simply flip the < to >
        g = G.copy()
        trues = [0] * G.shape[0]
        for i in range(G.shape[0]):
           trues[i] = np.random.uniform(0,1) > ((1-p)**weights[i])
        g['trues'] = trues
        g = g.loc[g['trues'] == True]
    elif weight == False:
        g = G.copy().loc[np.random.uniform(0,1,G.shape[0]) < p]
                

    # Step 3. Construct reverse reachable set of the random source node
    new_nodes, RRS0 = [source], [source]   
    while new_nodes:
        
        # Limit to edges that flow into the source node
        temp = g.loc[g['target'].isin(new_nodes)]

        # Extract the nodes flowing into the source node
        temp = temp['source'].tolist()

        # Add new set of in-neighbors to the RRS
        RRS = list(set(RRS0 + temp))

        # Find what new nodes were added
        new_nodes = list(set(RRS) - set(RRS0))

        # Reset loop variables
        RRS0 = RRS[:]

    return(RRS)
    

def ris(G,k,p=0.5,mc=1000, weight = False):    
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of RRSs to generate
    Return: A seed set of nodes as an approximate solution to the IM problem
    """
    
    # Step 1. Generate the collection of random RRSs
    start_time = time.time()
    R = [get_RRS(G=G,p=p,weight=weight) for _ in range(mc)]

    # Step 2. Choose nodes that appear most often (maximum coverage greedy algorithm)
    SEED, timelapse = [], []
    for _ in range(k):
        
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)
        
        # Remove RRSs containing last chosen seed 
        R = [rrs for rrs in R if seed not in rrs]
        
        # Record Time
        timelapse.append(time.time() - start_time)
    
    return(sorted(SEED),timelapse)