# This file contains functions for analyzing elections and clustering ballots.

# An elections is represented as a dictionary matching ballots to weights.
#   For example `{(1,3,4):5, (1): 7}` is the election where $5$ people cast the ballot $(1,3,4)$
#   while $7$ people cast the bullet vote for candidate $1$.  
#   The candidates are always named $1,2,...n$.

# A clustering of an election means a partition its ballots.  
#   Each cluster (each piece of the partition) is itself an election, 
#   so the clustering is represented as a list of elections.  

import csv
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import pandas as pd
import seaborn as sns
from itertools import permutations, combinations, chain
import more_itertools
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
import scipy as sp
from scipy import sparse
from scipy.stats import gaussian_kde
#import sknetwork as skn
from functools import partial
#from pyclustering.cluster.kmedians import kmedians as pyclust_kmedians
#from pyclustering.utils.metric import distance_metric, type_metric
from collections import Counter
from typing import Optional
from numpy.typing import NDArray

# Helper function for csv_parse
# (converts the ballot rows to ints while leaving the candidates as strings)
def convert_row(row):
    return [int(item) if item.isdigit() else item for item in row]


def csv_parse(filename):
    """
    Returns a tuple (num_cands, election, cand_list, ward) obtained from parsing the format in which Scottish election data is stored.
        
    The returned election is a dictionary matching ballots to weights.  For example {(1,3,4):5, (1): 7} is the election where 5 people cast the ballot (1,3,4) while 7 people cast the bullet vote for candidate 1.
    
    The candidates are coded 1,2,...n in the ballots of the returned election.  The returned cand_list tells the corresponding candidate names and their parties. 

    Args:
        filename : name of file (.csv or .blt) containing the Scottish election data.
    
    Returns:
        tuple: num_cands, election, cand_list, ward.
    """

    data = []
    with open(filename, "r", encoding = "utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # This just removes any empty strings that are since
            # we don't need to preserve column location within a row
            filtered_row = list(filter(lambda x: x != "", row))
            data.append(convert_row(filtered_row))

    num_cands = data[0][0]

    election_dict = dict()
    for row in data[1 : -num_cands - 1]:
        n_ballots = row[0]
        ballot = tuple(row[1:])
        election_dict[ballot] = n_ballots

    cand_list = []
    for cand in data[-num_cands - 1 : -1]:
        cand_name_list = cand[1].split()
        cand_first = " ".join(cand_name_list[:-1])
        cand_last = cand_name_list[-1]
        cand_party = cand[2]
        cand_list.append((cand_first, cand_last, cand_party))

    ward = data[-1][0]

    return num_cands, election_dict, cand_list, ward

def party_abrevs(cand_names):
    """
    Inputs the cand_list returned by csv_parse
    Returns a corresponding list of the party abreviations.     
    """
    to_return = []
    for cand in cand_names:
        party = cand[2]
        # Find the start and end indices of the parentheses
        start_idx = party.find('(')
        end_idx = party.find(')')
        
        # If parentheses are found, extract the text inside them and the rest of the string
        if start_idx != -1 and end_idx != -1:
            A = party[start_idx + 1:end_idx]
            B = party[:start_idx].strip() + party[end_idx + 1:].strip()
            to_return.append(A)
        else:
            to_return.append('')
    return to_return

# Helper function for Summarize_election
def print_color(text,n): # print the text in the color associated to the integer index n.
    """
    Helper function for Summarize_election
    """
    black_code = "\033[00m"
    color_code = f"\033[{91+n}m"
    print(color_code,text,black_code)    

def Summarize_election(election, clusters=None, size=10):
    """
    Prints basic data about the given election including num candidates, num ballots, num distinct ballots, average ballot length, and the 10 (or any number) most often cast ballots.

    If a clustering is also given, then it also prints this data saparately for each cluster and color codes by cluster the list of most commonly cast ballots.  

    Args:
        election : a dictionary matching ballots to weights (# times cast).
        clusters : a clustering (list of elections) that partition the ballots of the election. 
        size : The desired length of the list of the most commonly cast ballots.
    """
    all_ballots = [ballot for ballot in election.keys() if election[ballot]>0]
    num_ballots = len(all_ballots)
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    num_cands = len(candidates)
    num_voters = sum([election[ballot] for ballot in all_ballots])
    mu = sum([len(ballot)*election[ballot] for ballot in election.keys()])/sum(election.values())
    print(f"This election has: {num_cands} candidates, {num_voters} ballots, {num_ballots} distinct ballots, {round(mu,2)} avg ballot length.")

    if not clusters==None:
        for cluster_num in range(len(clusters)):
            cluster = clusters[cluster_num]
            all_ballots_c = [ballot for ballot in cluster.keys() if cluster[ballot]>0]
            num_ballots_c = len(all_ballots_c)
            num_voters_c = sum([cluster[ballot] for ballot in all_ballots_c])
            mu_c = sum([len(ballot)*cluster[ballot] for ballot in cluster.keys()])/sum(cluster.values())           
            print_color(f"CLUSTER {cluster_num+1}: {num_voters_c} ballots, {num_ballots_c} distinct ballots, {round(mu_c,2)} avg ballot length.",cluster_num)
            
    print("Top ballots:")
    ls = sorted(set(election.values()))
    count = 0
    broken = False
    while not broken:
        val = ls.pop()        
        bs = [ballot for ballot in all_ballots if election[ballot]==val]
        for ballot in bs:
            if clusters == None:
                color = -91 # black
            else:
                assignments = [n for n in range(len(clusters)) if ballot in clusters[n].keys() and clusters[n][ballot]>0]
                if len(assignments)==1:
                    color = assignments[0]
                else:
                    color = -84 # black outliner to indicate multiple clusters
            print_color(f"\t {val} votes for {ballot}.", color)
            count +=1
            if count>size:
                broken = True
                break

def Plot_ballot_lengths(clusters, num_cands = 'Auto', filename=None, dpi = 600):
    """
    Plots a histogram of the ballot lengths for the given election or clustering.

    If a clustering is given instead of an election, it superimposes histrograms for each cluster.  

    Args:
        clusters : either an election (a dictionary matching ballots to weights) or a clustering (a list of elections).
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        filename : to save the plot.
    """
    if type(clusters)==dict:
        clusters = [clusters]
    k = len(clusters)
    if num_cands == 'Auto':        
        all_ballots = [x for cluster in clusters for x in cluster] 
        num_cands = max([item for ranking in all_ballots for item in ranking])

    X = np.arange(num_cands)
    width = .7/k
    palat = ['grey','purple','tomato','orange','b','c','g', 'r', 'm', 'y', 'k']
    fig, ax = plt.subplots()

    for clust in range(k):
        Y = np.zeros(num_cands)
        for ballot, weight in clusters[clust].items():
            Y[len(ballot)-1]+=weight
        Y = Y/sum(clusters[clust].values())
        ax.bar(X+clust*width,Y, width=width, label = f"Cluster {clust+1}", color = palat[clust])
    ax.set_title('Ballot lengths')
    ax.set_xlabel('ballot length')
    plt.xticks(X+(width*(k-1)/2) ,X+1)
    plt.legend()
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi=dpi)

def Borda_vector(ballot_or_election,num_cands='Auto', borda_style='pes', start = 0):
    """
    Returns the Borda vector of the given ballot or election.  The "Borda vector of an election" means the (weighted) sum of the Borda vectors of its ballots.
    
    Set borda_style = 'pes' for  to use the "pessimistic" convenction that missing candidates receive the minimum of the scores they'd have had if they'd been ranked.
    
    Set borda_style = 'avg' for  to use the "Averaged" convenction that missing candidates receive the average of the scores they'd have had if they'd been ranked

        
    Args:
        ballot_or_election : a single ballot (tuple) or an election (dictionary matching ballots to weights)
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it, but only if an election is given, since a single ballot isn't enough to determine num_cands.
        borda_style : choice of {'pes', 'avg'}
        start : the lowest score awarded; for example, set start=1 if you want a full ballot to award {1,2,...,num_cands} points.
     
    Returns:
        the Borda vector (np.array) of the given ballot or election.                
    """
    # Borda vector of a ballot
    if type(ballot_or_election) == tuple:
        if num_cands=='Auto':
            raise Exception("A single ballot is not enough to determine the number of candidates.")
        ballot = ballot_or_election
        to_return = [0 for _ in range(num_cands)]
        for position in range(len(ballot)):
            candidate = ballot[position]
            to_return[candidate-1] = num_cands-position-1+start
        if borda_style=='avg':
            missing_cands = set(range(1,num_cands+1))-set(ballot)
            for candidate in missing_cands:
                to_return[candidate-1] = (len(missing_cands)-1)/2+start
            
    # Borda vector of an election
    else:
        election = ballot_or_election
        if num_cands == 'Auto':
            num_cands = max([item for ranking in election.keys() for item in ranking])

        to_return = [0 for _ in range(num_cands)]
        for ballot, ballot_weight in election.items():
            for position in range(len(ballot)):
                candidate = ballot[position]
                to_return[candidate-1] += ballot_weight*(num_cands-position-1+start)
            if borda_style=='avg':
                missing_cands = set(range(1,num_cands+1))-set(ballot)
                for candidate in missing_cands:
                    to_return[candidate-1] += ballot_weight*((len(missing_cands)-1)/2+start)
    
    return np.array(to_return)

def Borda_dist(CA, CB, num_cands = 'Auto', borda_style='pes', order = 1):
    """
    Returns the L^p distance between the Borda vectors of the given pair of ballots or elections,
        where p is called the order (for example, order=2 is the Euclidean distance).
    
    Set borda_style = 'pes' for  to use the "pessimistic" convenction that awards n-k points to the candidate in position k, and awards zero points for missing candidates.
    
    Set borda_style = 'avg' for  to use the "averaged" convenction that every ballot awards exactly 1+2+...+n Borda points; this is achieved for a short ballot by dividing the unawarded points equally among the missing candidates.
    
    Args:
        CA, CB : a pair of ballots (tuples) or elections (dictionaries matching ballots to weights).
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it,
                    but only if an election is given, since a ballot pair isn't enough to determine num_cands.
        borda_style : choice of {'pes', 'avg'}
        order : the choice of p with resepct to which the L^p distance is computed.
    
    Returns:
        the L^p distance between the Borda vectors.                
    """
    if num_cands == 'Auto':
        if type(CA) == tuple:
            raise Exception("A single pair of ballot is not enough to determine the number of candidates.")
        else:
            all_ballots = list(CA.keys())+list(CB.keys())
            num_cands = max([item for ranking in all_ballots for item in ranking])
            
    VA = Borda_vector(CA, num_cands=num_cands, borda_style=borda_style)
    VB = Borda_vector(CB, num_cands=num_cands, borda_style=borda_style)
    return (1/2)*np.linalg.norm(VA - VB,ord=order)

# Helper function for Plot_clusters
def Candidate_matrix(election, num_cands = 'Auto'):
    """
    Helper function for Plot_clusters   
    """
    # Creates a matrix M such that M[i-1][c-1] is the number of ith place votes received by candidate c.
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    
    to_return = np.zeros([num_cands,num_cands])
    for ballot in election.keys():
        ballot_weight = election[ballot]
        for ballot_position in range(len(ballot)):
            candidate = ballot[ballot_position]
            to_return[ballot_position][candidate-1] += ballot_weight
    return to_return

def Plot_clusters(clusters, method = 'Borda', borda_style='pes', num_cands = 'Auto',
                  order = 'Auto', title = 'Auto', palat = 'Auto', filename=None, dpi=600, ax = None):
    """
    Displays a bar plot that helps visualize the given election or clustering.

    Args:
        election: either an election (a dictionary matching ballots to weights) or a clustering (a list of elections).
        method: either 'Borda' for a Borda plot, or 'Mentions' for a stacked mentions plot.
        borda_style: choice of {'pes', 'avg'}, which is passed to Borda_vector.
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        order : Set order='Auto' to order the candidates by deceasing Borda scores in the first cluster.  Set say order=[3,2,4,1] to order the candidates according to the given list. 
        title : the title of the plot.  Set to 'Auto' to use a default title.
        filename : to save the plot.
        palat: a list of colors to use for the clusters.  If 'Auto', a default palette is used.
        ax: the axes to plot on.  If None, a new figure and axes are created.
    """
    if type(clusters)==dict:
        clusters = [clusters]
    k = len(clusters)
    if num_cands == 'Auto':        
        all_ballots = [x for cluster in clusters for x in cluster] 
        num_cands = max([item for ranking in all_ballots for item in ranking])
    if method=='Borda':
        Scores = [Borda_vector(clusters[n], num_cands=num_cands, borda_style=borda_style) for n in range(k)]
    else:
        Scores = [Candidate_matrix(clusters[n], num_cands = num_cands) for n in range(k)]

    if type(order)==list:
        perm = [x-1 for x in order]
    if order=='Auto': # Order candidates by Borda scores of first cluster
        perm = np.flip(np.argsort(np.array(Borda_vector(clusters[0], num_cands=num_cands))))
    if type(order)==list or order=='Auto':
        Ordered_candidates = []
        if method == 'Borda':
            Ordered_scores = [np.zeros(num_cands) for _ in range(k)]
        else:
            Ordered_scores = [np.zeros([num_cands,num_cands]) for _ in range(k)]
        for cand in range(num_cands):
            Ordered_candidates.append(perm[cand]+1)
            for clust in range(k):
                if method == 'Borda':
                    Ordered_scores[clust][cand] = Scores[clust][perm[cand]]
                else:
                    for ballot_position in range(num_cands):
                        Ordered_scores[clust][ballot_position,cand] = Scores[clust][ballot_position,perm[cand]]
    else:
        Ordered_candidates = list(range(1,num_cands+1))
        Ordered_scores = Scores

    if palat == 'Auto':
        palat = ['grey','purple','tomato','orange','b','c','g', 'r', 'm', 'y', 'k']
    r = np.arange(num_cands)
    width = 0.7/k
    bottoms = [np.zeros(num_cands) for _ in range(k)]
    if ax is None:
        fig, ax = plt.subplots()
    
    for clust in range(k):
        if method == 'Borda':
            pA = ax.bar(r + clust*width, Ordered_scores[clust], label=f"Cluster {clust+1}", color = palat[clust],
                width = width, edgecolor = 'black')
        else:
            for shade in range(1,num_cands+1):
                Shade_scores = Ordered_scores[clust][shade-1]
                label = f"Cluster {clust+1}" if shade==1 else None
                pA = ax.bar(r+clust*width, Shade_scores, width=width, bottom = bottoms[clust], 
                            color = (palat[clust],1/shade), label=label)
                bottoms[clust] += Shade_scores

    if title == 'Auto':
        title = 'Borda Scores of Candidates by Cluster' if method == 'Borda' else 'Candidate Mentions Stacked by Ballot Position'
    ax.set_title(title)

    #ax.set_xlabel('Candidate')
    ax.set_xticks(r + (width*(k-1))/2)
    ax.set_xticklabels(Ordered_candidates)
    ax.legend()
    if ax is None:
        if filename == None:
            plt.show()
        else:
            plt.savefig(filename, dpi=dpi)

def HH_proxy(ballot,num_cands, flatten = True):
    """
    Returns the head-to-head proxy vector of the given ballot.
    If flatten is False, the returned proxy is a matrix of head-to-head comparisons.
    Otherwise, it returns a vector with one entry for each pair of candidates ordered in the natural way; namely {(1,2),(1,3),...,(1,n),(2,3),...}.  
    The entries lie in {-1,0,1} depending on whether the lower-indexed candidate {looses, ties, wins} the head-to-head comparison. 

    Args:
        ballot: a single ballot (tuple)
    
    Returns:
        The head-to-head proxy vector (np.array)
    """
    M = np.zeros([num_cands,num_cands])
    for x,y in combinations(ballot,2):
        M[x-1,y-1] = 1
        M[y-1,x-1] = -1
    for x in ballot:
        for y in set(range(1,num_cands+1)) - set(ballot): # candidates missing from the ballot
            M[x-1,y-1] = 1
            M[y-1,x-1] = -1
    if not flatten:
        return M
    else:
        to_return = []
        for x,y in combinations(range(num_cands),2):
            to_return.append(M[x,y])
        return np.array(to_return)

def HH_dist(ballot1, ballot2, num_cands, order = 1):
    """
    Returns the L^p distance between the head-to-head proxy vectors of the given pair of ballots, where p is called the order.

    Args:
        ballot1 : ballot (tuple)
        ballot2 : ballot (tuple)
        num_cands : the number of candidates
        order : the choice of p with respect to which the L^p distance is computed
    
    Returns:
        The L^p distance between the proxy vectors.   
    """
    H1 = HH_proxy(ballot1, num_cands=num_cands)
    H2 = HH_proxy(ballot2, num_cands=num_cands)
    return (1/2)*np.linalg.norm(H1-H2,ord=order)

def Candidate_dist_matrix(election, num_cands = 'Auto', method = 'borda', borda_style='avg', trunc = None):
    """ 
    Returns a symmetric matrix whose (i,j) entry is one of these measurements of their 'distance':
    method = 'successive': the portion of ballots on which candidates i & j don't appear next to each other.
    method = 'coappearances': the portion of ballots on which candidates i & j don't both appear.
    method = 'borda' : the average diference in borda points that ballots award to candidates i & j
    method = 'borda_completion' : the average over the completions of the ballots of the diference in the borda points awarded to candidates i & j

    Args
        election : dictionary matching ballots to weights
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        method : one of {'successive', 'coappearances', 'borda', 'borda_completion'}
        borda_style : choice of {'pes', 'avg'}, which is passed to Borda_vector.  Only use for method == 'borda'.
        trunc : truncate all ballots at this position before applying the method.
    """
    if method == 'borda_completion':
        borda_style = 'avg' # borda_completion only makes sense with the averaged borda style.
        
    num_cands = max([item for ranking in election.keys() for item in ranking])
    
    if trunc == None:
        trunc = num_cands

    if method == 'successive' or method == 'coappearances':
        M = np.full((num_cands,num_cands),fill_value=1.0)
        for t in range(num_cands):
            M[t,t]=0.0
        increment = 1/sum(election.values())

    if method == 'successive':
        for ballot, weight in election.items():
            trunc_ballot = ballot[:trunc]
            for ballot_position in range(len(ballot)-1):
                c1 = ballot[ballot_position]-1
                c2 = ballot[ballot_position+1]-1
                M[c1,c2] -= increment*weight
                M[c2,c1] -= increment*weight
    elif method == 'coappearances':
        for ballot,weight in election.items():
            trunc_ballot = ballot[:trunc]
            for a,b in combinations(trunc_ballot,2):
                M[a-1,b-1] -=increment*weight
                M[b-1,a-1] -=increment*weight
    elif method == 'borda' or method == 'borda_completion':
        M = np.zeros([num_cands,num_cands])
        for ballot, weight in election.items():
            trunc_ballot = ballot[:trunc]
            num_missing = num_cands - len(ballot)
            v = Borda_vector(ballot, num_cands=num_cands, borda_style=borda_style)
            for i in range(num_cands):
                for j in range(num_cands):
                    M[i,j] += np.abs(v[i]-v[j])*weight
                    if method == 'borda_completion' and v[i]==v[j] and i!=j:
                        M[i,j] += weight*(num_missing+1)/3
    else:
        raise Exception("method must be one of {'successive', 'coappearances', 'borda', 'borda_completion'.")

    return M/sum(election.values())

def Candidate_MDS_plot(election, method = 'borda_completion', borda_style='avg', num_cands = 'Auto', dimension = 2, trunc = None, 
                       n_init = 500, metric = True, marker_scale = .5,
                       party_names = None, party_colors = None, filename = None, return_data = False, dpi = 600):
    """
    Prints a multidimensional scaling (MSD) plot of the candidates, labeled by party.  Markers are sized by number of first place votes.
    The "distance" it approximates is one of the following:
    method = 'successive': the portion of ballots on which candidates i & j don't appear next to each other.
    method = 'coappearances': the portion of ballots on which candidates i & j don't both appear.
    method = 'borda' : the average diference in borda points that ballots award to candidates i & j
    method = 'borda_completion' : the average over the completions of the ballots of the diference in the borda points awarded to candidates i & j


    Args
        election : dictionary matching ballots with weights.
        method : choice of {'successive', 'coappearances', 'borda', 'borda_completion'}
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        dimension : choice of {1,2,3} for dimension of MDS plot.
        trunc : truncate all ballots at this position before applying the method.
        n_init : The number of times the SMACOF algorith will run with different initialializations.
        metric : set to True to use metric MDS, or False for non-metric MDS.  Since the 'borda' and 'borda_completions' methods are metrics, metric=True is appropriate for these. 
        party_names : list of strings used as annotation labels.
        party_colors : 'Auto', None, or list of colors.  Only use 'Auto' of party_names is provided.
        filename : set to None if you don't want to save the plot.
        return_data: useful if you want to know the projection error, or for constructing multiple MDS plots of the same election using a common projection.

    Returns:
        projections, error (if return_data is set to True)
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    M = Candidate_dist_matrix(election, num_cands, method = method, borda_style=borda_style, trunc = trunc)

    # compute c = marker colors
    if party_colors == None:
        c = ['blue' for _ in range(num_cands)]
    elif party_colors == 'Auto':
        D = {'SNP':'#FFE135', 'Lab': '#E32636', 'Con':'#0F4D92','LD':'#FF9933','Gr':'#4CBB17', 'Ind': '#008B8B'}
        c = []
        for party in party_names:
            c.append(D[party] if party in D.keys() else 'black')
    else:
        c = party_colors

    # compute s = marker sizes
    s = [0 for _ in range(num_cands)]
    for ballot,weight in election.items():
        s[ballot[0]-1]+=weight
    s = marker_scale*np.array(s)

    # compute projections
    model = MDS(n_components=dimension, dissimilarity='precomputed', n_init=n_init, metric=metric)
    projections = model.fit_transform(M)
    error = model.stress_
    X = np.array([p[0] for p in projections])
    Y = np.array([p[1] for p in projections]) if dimension>1 else np.array([0 for _ in range(num_cands)])

    if dimension<3:
        fig, ax = plt.subplots()
        ax.scatter(X,Y, c = c, s = s, alpha=.5)

        x_margin = (max(X) - min(X)) * 0.2  # 20% margin
        plt.xlim(min(X) - x_margin, max(X) + x_margin)
        if dimension==2:
            y_margin = (max(Y) - min(Y)) * 0.2  # 20% margin
            plt.ylim(min(Y) - y_margin, max(Y) + y_margin)
        ax.grid(False)
        ax.axis('off')

    else:
        Z = np.array([p[2] for p in projections])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X,Y,Z, c=c, s=s)
        ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    if not party_names == None:
        for count in range(num_cands):
            if dimension == 3:
                ax.text(X[count],Y[count],Z[count], f" {count+1}({party_names[count]})")
            else:
                ax.annotate(f" {count+1}({party_names[count]})", xy=(X[count], Y[count]))

    if filename != None:
        plt.savefig(filename, dpi=dpi)
    plt.show()

    if return_data:
        return projections, error
    
# Helper function for Group_candidates
def List_merge(L,i,j): # Merges entries i and j of the given list.
    """
    Helper function for Group_candidates.   
    """
    n = len(L)
    to_return = []
    for x in range(n-1):
        offset = 0 if x<j else 1
        if x<i:
            to_return.append(L[x])
        elif x==i:
            to_return.append(L[i].union(L[j]))
        else:
            to_return.append(L[x+offset])
    return to_return

# Helper function for Group_candidates
def List_merge(L,i,j): # Merges entries i and j of the given list.
    """
    Helper function for Group_candidates.   
    """
    n = len(L)
    to_return = []
    for x in range(n-1):
        offset = 0 if x<j else 1
        if x<i:
            to_return.append(L[x])
        elif x==i:
            to_return.append(L[i].union(L[j]))
        else:
            to_return.append(L[x+offset])
    return to_return

def Group_candidates(election, num_cands = 'Auto', method = 'borda_completion', borda_style='avg', trunc = None, 
                     link = 'avg', verbose = True, return_all = False):
    """
    Prints the steps of repeatedly grouping candidates via agglomerative clustering
    using Candidate_dist_matrix as the pairwise distances.

        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        method : one of {'successive', 'coappearances', 'borda', 'borda_completion'} for the method passed to Candidate_dist_matrix.
        borda_style : one of {'avg', 'pes'} for the type of Borda count to use (only if method is 'borda').
        trunc : truncate all ballots at this position before applying the method.
        link : one of {'min', 'avg', 'max'} for single, averaged or complete linkage clustering.
        verbose : set to True to print the steps of the algorithm.

        (if return_all==False) returns final bipartition of candidates as a list of sets [S1,S2]
        (if return_all==True) returns all steps of the algorithm as a list of lists of sets [[S1,S2,...], [S1,S2,...], ...] 
        where each inner list is a step in the algorithm.
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    M = Candidate_dist_matrix(election, num_cands, method = method, borda_style = borda_style, trunc = trunc)
    L = [{n} for n in range(1,num_cands+1)]
    to_return = [L]
    if verbose:
        print(L)
    while len(L)>1:
        best_val = np.inf
        best_pair = (np.nan,np.nan)
        for i in range(len(L)):
            for j in range(i+1,len(L)):
                comparisons = [M[x-1,y-1] for x in L[i] for y in L[j]]
                if link == 'avg':
                    score = np.mean(comparisons)
                elif link == 'max':
                    score = max(comparisons)
                elif link == 'min':
                    score = min(comparisons)
                else:
                    raise Exception("link must be 'avg' or 'min' or 'max'.")
                if score<best_val:
                    best_val = score
                    best_pair = (i,j)
        L = List_merge(L,best_pair[0],best_pair[1])
        if verbose:
            print(L)
        to_return.append(L)
    if return_all:
        return to_return[0:-1]  # Exclude the final step which is just one set with all candidates
    else:
        return to_return[-2] 

def kmeans(election, k=2, proxy='Borda', borda_style='pes', n_init=200, return_centroids=False):
    """
    Returns the clustering obtained by applying the k-means algorithm to the proxies of the ballots.

    Args:
        election : dictionary matching ballots with weights.
        k : the number of clusters desired.
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'pes', 'avg'}, which is passed to Borda_vector (only if proxy == 'Borda') 
        n_init : the algorithm runs n_init independent times with different starting centers each time, and outputs the clustering that has the best score from all the runs.
        return_centroids : set to True if you want it to also return the centroids of the returned clustering.

    Returns:
        if return_centroids == False: returns a clustering (list of elections).
        if return_centroids == True: returns a tuple (clustering, centroids).
    """
    all_ballots = list(election.keys())
    num_ballots = len(all_ballots)
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    num_cands = len(candidates)
    sample_weight = np.array([election[ballot] for ballot in all_ballots])
    if proxy=='Borda':
        X = np.array([Borda_vector(ballot, num_cands=num_cands, borda_style=borda_style) 
                      for ballot in all_ballots])
    else:
        X = np.array([HH_proxy(ballot,num_cands=num_cands) for ballot in all_ballots])
    
    model = KMeans(n_clusters=k, n_init=n_init).fit(X,sample_weight=sample_weight)
    labels = model.labels_
    centroids = model.cluster_centers_
    
    C = [dict() for _ in range(k)]
    for count in range(len(all_ballots)):
        ballot = all_ballots[count]
        C[labels[count]][ballot]=election[ballot]
    if return_centroids:
        return C, centroids
    else:
        return C
    
# Peter's implementation of PAM
def __build_phase(
    distance_matrix: NDArray, wt_vector: NDArray, k: int
) -> tuple[set[int], set[int]]:
    """
    Constructs the initial medoids for the PAM algorithm. And computes the
    distance from each point to its nearest and second nearest medoids.
    https://www.cs.umb.edu/cs738/pam1.pdf

    Args:
        distance_matrix (NDArray): A square matrix of pairwise distances between points.
        wt_vector (NDArray): A vector of weights for each point.
        k (int): Number of clusters (medoids) to find.

    Returns:
        selected_medoids (set[int]): The indices of the selected medoids.
        unselected_objects (set[int]): The indices of the unselected objects.
    """
    dim = distance_matrix.shape[0]

    weighted_total_dist_to_col = (distance_matrix * wt_vector[:, np.newaxis]).sum(
        axis=0
    )
    # The first medoid is the one which minimizes the weighted
    # distance from all other points to itself
    first_medoid = int(np.argmin(weighted_total_dist_to_col))
    selected_medoids = set({first_medoid})
    unselected_objects = set(range(dim)) - selected_medoids

    dissimilarities = distance_matrix[:, first_medoid].copy()

    while len(selected_medoids) < k:
        best_gain = -np.inf
        best_medoid_candidate = None

        for candidate in unselected_objects:
            gain = np.dot(
                np.maximum(dissimilarities - distance_matrix[:, candidate], 0),
                wt_vector,
            )
            if gain > best_gain:
                best_gain = gain
                best_medoid_candidate = candidate

        assert best_medoid_candidate is not None
        selected_medoids.add(best_medoid_candidate)
        unselected_objects.remove(best_medoid_candidate)

        new_distances = distance_matrix[:, best_medoid_candidate]

        dissimilarities = np.minimum(dissimilarities, new_distances)

    return selected_medoids, unselected_objects


def __find_best_swap(
    distance_matrix: NDArray,
    wt_vector: NDArray,
    selected_medoids: set[int],
    unselected_objects: set[int],
    dissimilarities: NDArray,
    second_dissimilarities: NDArray,
    nearest_medoids: NDArray,
) -> tuple[float, tuple[int, int]]:
    """
    Given the current set of selected medoids and unselected objects, finds the best single swap
    between medoids and objects that will improve the clustering.
    https://www.cs.umb.edu/cs738/pam1.pdf


    Args:
        distance_matrix (NDArray): A square matrix of pairwise distances between points.
        wt_vector (NDArray): A vector of weights for each point.
        selected_medoids (set[int]): The indices of the selected medoids.
        unselected_objects (set[int]): The indices of the unselected objects.
        dissimilarities (NDArray): The dissimilarities between each point and its nearest medoid.
        second_dissimilarities (NDArray): The dissimilarities between each point and its second nearest medoid.
        nearest_medoids (NDArray): The indices of the nearest medoid for each point.

    Returns:
        best_swap_cost (float): The cost of the best swap. Will be 0 if no beneficial swap is found.
        best_pair (tuple[int, int]): The indices of the medoid and object that will be swapped.
            Will be (-1, -1) if no beneficial swap is found.
    """
    assert (
        selected_medoids.intersection(unselected_objects) == set()
    ), f"Selected medoids and unselected objects must be disjoint, but found {selected_medoids.intersection(unselected_objects)} in both"

    best_swap_cost = 0.0  # Negative marginal cost indicates improvement
    best_pair = (-1, -1)

    for medoid in selected_medoids:
        indicator_idx_belongs_to_medoid = nearest_medoids == medoid
        not_indicator_idx_belongs_to_medoid = ~indicator_idx_belongs_to_medoid

        for candidate in unselected_objects:
            distances_to_candidate = distance_matrix[:, candidate]

            # Contribution when the medoid is the nearest neighbor of the candidate
            contribution_near = (
                np.minimum(
                    distances_to_candidate[indicator_idx_belongs_to_medoid],
                    second_dissimilarities[indicator_idx_belongs_to_medoid],
                )
                - dissimilarities[indicator_idx_belongs_to_medoid]
            )

            # Now compute the contribution when another medoid is the nearest neighbor
            difference = (
                distances_to_candidate[not_indicator_idx_belongs_to_medoid]
                - dissimilarities[not_indicator_idx_belongs_to_medoid]
            )
            contribution_far = np.minimum(difference, 0.0)

            swap_cost = (
                contribution_near * wt_vector[indicator_idx_belongs_to_medoid]
            ).sum() + (
                contribution_far * wt_vector[not_indicator_idx_belongs_to_medoid]
            ).sum()

            if swap_cost < best_swap_cost:
                best_swap_cost = swap_cost
                best_pair = (medoid, candidate)

    return best_swap_cost, best_pair


def __compute_dissimilarities(distance_matrix, selected_medoids):
    """
    Computes the dissimilarities between each point and the selected medoids.

    Args:
        distance_matrix (NDArray): A square matrix of pairwise distances between points.
        selected_medoids (set[int]): The indices of the selected medoids.

    Returns:
        dissimilarities (NDArray): The dissimilarities between each point and its nearest medoid.
        second_dissimilarities (NDArray): The dissimilarities between each point and its
            second nearest medoid.
        nearest_medoids (NDArray): The indices of the nearest medoid for each point.
    """
    dim = distance_matrix.shape[0]

    medoid_list = list(selected_medoids)
    distances_to_medians = distance_matrix[:, medoid_list]
    nearest_pos = np.argmin(distances_to_medians, axis=1)
    dissimilarities = distances_to_medians[np.arange(dim), nearest_pos]

    # Sorts the rows so that the second column is the second smallest distance
    partition = np.partition(distances_to_medians, 1, axis=1)
    second_dissimilarities = partition[:, 1]

    nearest_medoids = np.array([medoid_list[pos] for pos in nearest_pos])

    return dissimilarities, second_dissimilarities, nearest_medoids


def pam(
    distance_matrix: NDArray,
    k: int,
    max_iter: int = 100,
    weight_vector: Optional[NDArray] = None,
) -> tuple[list[int], list[int], float]:
    """
    Partitioning-Around-Medoids (PAM) using a pre-computed distance matrix.
    https://www.cs.umb.edu/cs738/pam1.pdf

    Args:
        distance_matrix (NDArray): A square matrix of pairwise distances between points.
        k (int): Number of clusters (medoids) to find.
        max_iter (int): Maximum number of iterations for the swap phase.
        weights (NDArray | None): Optional weights for each point. If None, all points
            are treated equally.

    Returns:
        medoids (NDArray): Indices of the selected medoids.
        nearest_medoids (NDArray): An assignment vector of indices to nearest medoid
        best_cost (float): The total cost of the clustering, defined as the sum of distances
            from each point to its nearest medoid, weighted by the provided weights
    """
    if k == 1:
        return ([0], [0] * distance_matrix.shape[0], 0.0)

    dim = distance_matrix.shape[0]
    if distance_matrix.shape[1] != dim:
        raise ValueError("distance_matrix must be square")

    wt_vector = (
        np.ones(dim) if weight_vector is None else np.asarray(weight_vector, float)
    )
    if (wt_vector < 0).any():
        raise ValueError("weights must be non-negative")

    wt_vector = wt_vector.squeeze()
    assert wt_vector.shape == (dim,)

    selected_medoids, unselected_objects = __build_phase(distance_matrix, wt_vector, k)

    dissimilarities, second_dissimilarities, nearest_medoids = (
        __compute_dissimilarities(distance_matrix, selected_medoids)
    )

    for _ in range(max_iter):
        best_swap_cost, best_pair = __find_best_swap(
            distance_matrix,
            wt_vector,
            selected_medoids,
            unselected_objects,
            dissimilarities,
            second_dissimilarities,
            nearest_medoids,
        )

        if best_swap_cost < 0 and best_pair != (-1, -1):
            medoid, candidate = best_pair

            selected_medoids.remove(medoid)
            selected_medoids.add(candidate)
            unselected_objects.add(medoid)
            unselected_objects.remove(candidate)

            dissimilarities, second_dissimilarities, nearest_medoids = (
                __compute_dissimilarities(distance_matrix, selected_medoids)
            )

        else:
            # No improvement found, exit the loop
            break

    return list(selected_medoids), nearest_medoids, sum(dissimilarities * wt_vector)

def Manhattan_dist(A,B):
    return sum(np.abs(A-B))

def kmedoids(election, k=2, proxy='Borda', borda_style='pes', return_medoids=False):
    """
    Returns the clustering obtained by applying the PAM k-medoid algorithm to the proxies of the ballots.

    Args:
        election : dictionary matching ballots with weights.
        k : the number of clusters desired.
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'pes', 'avg'}, which is passed to Borda_vector (only if proxy == 'Borda') 
        method : choice of {'pam','alternate'}.  The method 'pam' is more accurate, while 'alternate' is faster
        share_ties : set to True if you want the weight of any ballot that's equidistant to mulitple medoids to be shared between the corresponding clusters in the final iteration. This requires overlaid code because sklearn gives ties to the lowest-indexed cluster (which causes repeatability isses).  
        return_medoids : set to True if you want it to also return the medoids of the returned clustering.

    Returns:
        if return_medoids == False: returns a clustering (list of elections).
        if return_medoids == True: returns a tuple (clustering, medoids).
    """
    num_cands = max([item for ranking in election.keys() for item in ranking])

    # create a matrix whose rows are the proxies of the unique ballots 
    # and a dictionary matching each ballot type with its corresponding row in the matrix
    # and a reverse dictionary to match each row number of the matrix with a ballot
    # and a weights array.
    X = []
    ballot_to_row = dict()
    row_to_ballot = dict()
    weights = []
    counter = 0
    for ballot, weight in election.items():
        ballot_to_row[ballot]=counter
        row_to_ballot[counter]=ballot
        weights.append(weight)
        if proxy=='Borda':
            X.append(Borda_vector(ballot, num_cands=num_cands, borda_style=borda_style))
        else:
            X.append(HH_proxy(ballot,num_cands=num_cands))
        counter +=1
    M = pairwise_distances(X, metric="manhattan")
    medoid_indices, nearest_medoids, best_cost = pam(M, k=k, weight_vector=weights)
    medoid_ballots = [row_to_ballot[index] for index in medoid_indices]
    cluster_assignments = {ballot: medoid_indices.index(nearest_medoids[ballot_to_row[ballot]]) for ballot in election.keys()}
    # convert labels into a clustering (list of dictionaries)
    C = [dict() for _ in range(k)]    
    for ballot, weight in election.items():
        lab = cluster_assignments[ballot]
        C[lab][ballot]=C[lab].get(ballot,0)+weight

    if return_medoids:
        return C, medoid_ballots
    else:
        return C

def Clusters_from_centers(election, centers, proxy = 'Borda', borda_style = 'pes', order=1, centers_live_in_proxy_space = False):
    """
    Given the centers, this clusters the ballots according to their L^p closest center (where p=order).
    Return the clustering and the score (the sum of distances from each ballot to its closest center).
    ARGS:
        election: dictionary mapping ballots to weights
        centers: list of ballots or ballot proxies
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'pes', 'avg'}, which is passed to Borda_vector (only if proxy == 'Borda')  
        order: must be 1 or 2 (for the choice of p for L^p distance).  Use 1 for Manhattan, 2 for Euclidean.
        centers_live_in_proxy_space: True if the centers are in proxy space, False if they are in ballot space
    RETURNS:
        (summed distance, clustering)
        summed distance = sum of L^p distances from each ballot to its closest center
        clustering = dictionary mapping index to cluster.
    """
    num_cands = max([item for ranking in election.keys() for item in ranking])
    k = len(centers)

    C = {i:dict() for i in range(k)} # initialize clustering
    running_sum = 0
    for ballot, weight in election.items():
        if centers_live_in_proxy_space:
            if proxy == 'Borda':
                ballot_proxy = Borda_vector(ballot, num_cands, borda_style=borda_style)
            else:
                ballot_proxy = HH_proxy(ballot, num_cands)
            dists = [(1/2)*np.linalg.norm(ballot_proxy - center,ord=order) for center in centers]
        else:
            if proxy == 'Borda':
                dists = [Borda_dist(ballot, center, num_cands, borda_style=borda_style, order=order) for center in centers]
            else:
                dists = [HH_dist(ballot, center, num_cands, order=order) for center in centers]

        running_sum += weight*np.min(dists)
        clusts = [x for x in range(k) if dists[x]==np.min(dists)] # multi-valued argmin
        for clust in clusts:
            C[clust][ballot]=weight/len(clusts)
    return running_sum, C

def Random_clusters(election,k=2): # returns a random clustering of the ballots.
    """ 
    Returns the clustering obtained by performing a random partition of the ballots.
    The full weight if each ballot is put into a single randomly selected one of the clusters.

    Args:
        election : dictionary matching ballots with weights.
        k : the number of clusters desired.
    
    Returns:
        a clustering (list of elections).
    """
    C = [dict() for _ in range(k)]
    for ballot in list(election.keys()):
        die = random.randint(0,k-1)
        C[die][ballot] = election[ballot]
    return C

def Mallows_election(num_cands, num_clusters, centers, sizes, p=.5):
    """
    Mallows model for generating a random clustered election with complete ballots.

    Parameters:
        num_cands: number of candidates
        num_clusters: number of clusters
        centers: list of ballots that form the centers of the clusters (must be complete ballots)
        sizes: list sizes for each cluster
        p: parameter the geometric random variable that determines how many random adjacent swaps are made, starting from the center, to obtain each ballot.
    Returns:
        election, clustering
    """
    election = dict()
    clustering = [dict() for _ in range(num_clusters)]

    for i in range(num_clusters):
        for _ in range(sizes[i]):
            ballot = list(centers[i])
            num_swaps = np.random.geometric(p)
            for __ in range(num_swaps):
                swap_idx = np.random.randint(num_cands-1)
                x = ballot[swap_idx]
                y = ballot[swap_idx+1]   
                ballot[swap_idx] = y
                ballot[swap_idx+1] = x
            ballot = tuple(ballot)
            clustering[i][ballot] = clustering[i].get(ballot, 0) + 1
            election[ballot] = election.get(ballot, 0) + 1
    return election, clustering

def Clustering_closeness(election,C1,C2, num_cands = 'Auto', return_perm = False):
    """
    Returns the closeness of the given two clusterings, which means the portion of the total ballots for which the two partitions differ 
    (with respect to the best matching of one partition's pieces with the other partition's pieces)
    
    Args:
        election : a dictionary matching ballots to weights.
        C1 : a clustering (list of elections) 
        C2 : a clustering (list of elections) 
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        return_perm : If you wish for the best matching to also be returned.

    Returns:
        best_score, best_perm if return_perm else best_score
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    k = len(C1)
    if k != len(C2):
        raise Exception('C1 and C2 must same size.')
    
    perm_list = list(permutations(range(k)))
    perm_scores = dict()
    for perm in perm_list:
        score = 0
        for cluster_num in range(k):
            C1_piece = C1[cluster_num]
            C2_piece = C2[perm[cluster_num]]
            for ballot, weight in C1_piece.items():
                if ballot in C2_piece.keys():
                    score += np.abs(weight - C2_piece[ballot])
                else:
                    score += weight
        perm_scores[perm]=score/sum(election.values())
    
    best_score = min(perm_scores.values())
    best_perm = [perm for perm in perm_list if perm_scores[perm] == best_score][0]

    if return_perm:
        return (best_score,best_perm)
    else:
        return best_score

def Centroid_and_Medoid(C, num_cands = 'Auto', proxy='Borda', borda_style='pes', metric = 'Manhattan'):
    """ 
    Returns the centroid and medoid of the given election, C, which will typically be a single cluster.
    The returned centroid is a proxy, while the returned medoid is a ballot.
    
    Args:
        C : an election (typically a single cluster of an election)
        choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'pes', 'avg'}, which is passed to Borda_vector (only if proxy == 'Borda') 
        metric : choice of {'Euclidean', 'Manhattan'} for the distance function on the proxy vectors.

    Returns:
        tuple (centroid, medoid)
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in C.keys() for item in ranking])
    X = [] # list of proxies
    ballots = [] # list of ballots
    weights = []  # list of weights
    for ballot, weight in C.items():
        if proxy == 'Borda':
            X.append(Borda_vector(ballot, num_cands=num_cands, borda_style=borda_style))
        else:
            X.append(HH_proxy(ballot,num_cands=num_cands))
        weights.append(weight)
        ballots.append(ballot)

    weights = np.array(weights)
    X = np.array(X)
    if metric == 'Manhattan':
        similarities = manhattan_distances(X) 
    else:
        similarities = euclidean_distances(X)
    row_sums = [np.dot(row,weights) for row in similarities]
    medoid_index = np.argmin(row_sums)
    medoid = ballots[medoid_index]
    centroid = [np.dot(X[:,col_num],weights)
                    for col_num in range(len(X[0]))]
    centroid  = np.array(centroid)/sum(weights)
    
    return centroid, medoid

def Ballot_MDS_plot(election, clusters = None, num_cands = 'Auto', dimension = 2, n_init = 100, proxy='Borda', borda_style='pes',
                       threshold=10, label_threshold = np.inf, metric = 'Euclidean', 
                       party_names=None, filename=None, dpi = 600, 
                       projections = 'Auto', return_data = False, palat = 'Auto'):
    """
    Displays an MDS (multi-dimensional scaling) plot for the proxies of all of the ballots in the election that received at least the given threshold number of votes.
    If clusters is provided, they are colored by their cluster assignments; otherwise, by party of 1st place vote.
    
    Args:
        election : a dictionary matching ballots to weights.
        clusters : if a clustering is provided, it will color by cluster assignment.
        dimension : choice of {1,2,3} for the dimension of the MDS plot.
        n_init : The number of times the SMACOF algorith will run with different initialializations.
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'pes', 'avg'}, which is passed to Borda_vector (only used if proxy == 'Borda') 
        threshold : it ignores all ballots that were cast fewer than the threshold number of times.
        label_threshold : it labels all ballots that were cast at least the label_threshold number of times (set label_threshold=np.inf for no labeling)
        metric : choice of {'Euclidean', 'Manhattan'} for the proxy metric that's approximated.
        party_names : if provided, it will color by party of first place vote.
        filename : to save the plot.   
        projections: (optional) entering projections is useful for constructing multiple MDS plots of the same election using a common projection. 
        return_data: useful if you want to know the projection error, or for constructing multiple MDS plots of the same election using a common projection.
        palat : a list of colors for the clusters.  Set to 'Auto' to use the default color palette.
    Returns:
        projections, error (if return_data is set to True)
    """

    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])

    cluster_palat = ['grey','purple','brown','orange','b','c','g', 'r', 'm', 'y'] if palat == 'Auto' else palat
    party_palat_dic = {'SNP':'#FFE135', 'Lab': '#E32636', 'Con':'#0F4D92','LD':'#FF9933','Gr':'#4CBB17', 'Ind': '#008B8B'}

    ballots = []
    proxies = []
    weights = []
    colors = []

    for ballot, weight in election.items():
        if weight>=threshold:
            if proxy=='Borda':
                ballot_proxy = Borda_vector(ballot,num_cands=num_cands, borda_style=borda_style)
            else:
                ballot_proxy = HH_proxy(ballot,num_cands=num_cands)
            ballots.append(ballot)
            proxies.append(ballot_proxy)
            weights.append(weight)

            if party_names != None: # color by party of first place vote
                party = party_names[ballot[0]-1]
                color = party_palat_dic[party] if party in party_palat_dic.keys() else 'black'
                colors.append(color)
            elif clusters != None: # color by cluster assignment
                cluster_assigment = []
                for cluster_num in range(len(clusters)):
                    C = clusters[cluster_num]
                    if ballot in C.keys():
                        cluster_assigment.append(cluster_num)
                if len(cluster_assigment) == 0:
                    color = 'black'
                elif len(cluster_assigment) == 1:
                    color = cluster_palat[cluster_assigment[0]]
                else:
                    color = 'white'
                colors.append(color)
            else: # color everything black
                colors.append('black')

    if metric == 'Euclidean':
        similarities = euclidean_distances(proxies)
    else:
        similarities = manhattan_distances(proxies)
    if type(projections) == str: # if projections == 'Auto'
        model = MDS(n_components=dimension, dissimilarity='precomputed', n_init=n_init)
        projections = model.fit_transform(similarities)
        error = model.stress_
    else:
        error = None

    X = np.array([p[0] for p in projections])
    Y = np.array([p[1] for p in projections]) if dimension>1 else np.array([0 for _ in range(len(X))])

    if dimension<3:
        fig, ax = plt.subplots()
        ax.scatter(X,Y, s = weights, c = colors, alpha = .5)
        x_margin = (max(X) - min(X)) * 0.2  # 20% margin
        plt.xlim(min(X) - x_margin, max(X) + x_margin)
        if dimension == 2:
            y_margin = (max(Y) - min(Y)) * 0.2  # 20% margin
            plt.ylim(min(Y) - y_margin, max(Y) + y_margin)
        ax.grid(False)
        ax.axis('off')
    else:
        Z = np.array([p[2] for p in projections])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X,Y,Z, c=colors, s=weights)
        ax.set(xticklabels=[], yticklabels=[], zticklabels=[])        


    for count in range(len(proxies)):
        if weights[count]>label_threshold:
            if dimension == 3:
                ax.text(X[count],Y[count],Z[count], f"{ballots[count]}")
            else:
                ax.annotate(ballots[count], xy=(X[count], Y[count]))

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi = dpi)

    if return_data:
        return projections, error

# Borda proxy function that works with generalized ballots
# (Helper function for Slate_cluster_centers)
def Borda_proxy_gen(ballot, num_cands , borda_style='pes', start = 0):
    """
    Returns the Borda vector of the given (simple or generalized) ballot.
        
    Args:
        ballot : a simple or generalized ballot (tuple of integers or of sets of integers). 
        num_cands : the number of candidates. 
        borda_style : choice of {'pes', 'avg'}
        start : the lowest score awarded; for example, set start=1 if you want a full ballot to award {1,2,...,num_cands} points.
     
    Returns:
        the Borda vector (np.array) of the given generalized ballot.                
    """

    # append set of missing candidates to end of ballot
    ballot = list(ballot)
    missing_cands = set(range(1,num_cands+1))
    for c in ballot:
        S = c if type(c) == set else {c}
        for x in S:
            missing_cands.discard(x)
    if len(missing_cands) > 0:
        ballot.append(missing_cands)
    # compute Borda vector
    score_queue = list(range(start, start+num_cands))
    to_return = [0 for _ in range(num_cands)]
    for c in ballot:
        S = c if type(c) == set else {c}
        scores = [score_queue.pop() for _ in range(len(S))]
        points = np.mean(scores) if borda_style == 'avg' else min(scores)
        for x in S:
            to_return[x-1] = points

    return np.array(to_return)

# "centers" version of slate clustering that uses PAM to find the centers of the slates,
#  and then assigns ballots to their closest center

def Slate_cluster_centers(election, k):
    """
    Version of slate clustering that:
    1. clusters the candidates via PAM wrt d_B distances
    2. regards slates as generalized ballots with proxies in R^n (via Borda pessimistic embedding)
    3. and assigns each ballot to the closest slate.

    Returns slates, clustering
    """
    num_cands = max([item for ranking in election.keys() for item in ranking])
    
    # Step 1: cluster the candidates via PAM wrt d_B distances
    M = Candidate_dist_matrix(election, method = 'borda', borda_style='pes')
    medoid_indices, nearest_medoids, best_cost = pam(M, k=k)
    slates = []
    for medoid in medoid_indices:
        slate = set()
        for j, med in enumerate(nearest_medoids):
            if med == medoid:
                slate.add(j+1)
        slates.append(slate)
    slate_proxies = [Borda_proxy_gen((s,), num_cands, borda_style='pes') for s in slates]

    # step 2: assign each ballot to the closest slate
    C = [dict() for _ in range(k) ] # C[i] = cluster of ballots assigned to slate i
    for ballot, weight in election.items():
        ballot_vec = Borda_proxy_gen(ballot, num_cands, borda_style='pes')
        dists = [np.linalg.norm(ballot_vec - proxy, ord = 1) for proxy in slate_proxies]
        # share the weight if there are ties for closest slate
        min_dist = min(dists)
        for i, d in enumerate(dists):
            if d == min_dist:
                C[i][ballot] = C[i].get(ballot, 0) + weight / dists.count(min_dist)
    return slates, C

# Helper function for Slate_cluster
def blocs_from_slates(election, slates, num_cands='Auto', shared_ties = True, borda_style = 'pes'):
    """
    Returns the blocs (partition of the ballots) induced by the given slates (partition of the candidates).
    args:
        election : a dictionary mapping ballots (tuples) to counts (ints)
        slates : a list of sets of candidates, e.g. [{1,2},{3,4,5}, {6,7}]
        shared_ties : if True, ballots that tie for best slate are split evenly among their best slates.  If False, they are assigned to the lowest index cluster.
        borda_style : choice of {'pes', 'avg'}
    returns:
        a clustering, e.g. [C1, C2, C3] where each Ci is an election (dictionary mapping ballots to counts)
    """
    k = len(slates)
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])        
    C = {i:dict() for i in range(k)}

    for ballot, weight in election.items():
        proxy = Borda_vector(ballot, num_cands, borda_style=borda_style)
        scores = [np.mean([proxy[cand-1] for cand in slate]) for slate in slates]
        best = np.where(scores == np.amax(scores))[0] # indices of closest slates
        if shared_ties:
            for i in best:
                C[i][ballot] = C[i].get(ballot,0) + weight/len(best)
        else:
            i = min(best)
            C[i][ballot] = C[i].get(ballot,0) + weight
    return [C[i] for i in range(k)]

# helper function for Slate_cluster
def ballot_to_simplex_coords(ballot, slates, num_cands = 'Auto', method = 'borda'):
    """
    Returns the simplex coordinates of given ballot with respect to the given slates.
    args:
        ballot: a tuple of candidates, e.g. (1,2,3,4,5,6,7)
        slates : a list of sets of candidates, e.g. [{1,2},{3,4,5}, {6,7}]
        method : Use 'borda' for Borda accounting.  Use 'strong' or 'weak' for head-to-head accounting.
    returns:
        a list of lists of positive scores (one for each slate) that sum to 1.  Higher scores indicate a closer match to the slate.
    """
    if num_cands == 'Auto':
        num_cands = sum(len(slate) for slate in slates)
    k = len(slates)
    if method == 'borda':
        proxy = Borda_vector(ballot, num_cands, borda_style='pess')
        scores = [np.mean([proxy[cand-1] for cand in slate]) for slate in slates]
        return scores / sum(scores)
    else:
        proxy = HH_proxy(ballot, num_cands, flatten=False)
        alpha = np.zeros([k,k])
        for i,j in combinations(range(k), 2):
            l = [proxy[x-1,y-1] for x in slates[i] for y in slates[j]]
            wins = sum([1 for x in l if x == 1])
            losses = sum([1 for x in l if x == -1])
            ties = sum([1 for x in l if x == 0])
            if method == 'strong':
                alpha[i,j] = (wins + 0.5*ties)/ len(l)
            elif method == 'weak':
                alpha[i,j] = wins / (wins + losses) if wins+losses > 0 else .5
            else:
                raise ValueError("Method must be 'borda', 'strong' or 'weak'.")
            alpha[j,i] = 1 - alpha[i,j]
        to_return = []
        denom = k*(k-1)/2 # k-choose-2
        for i in range(k):
            to_return.append(sum([(alpha[i,j]) for j in range(k)])/denom)
        return np.array(to_return)

# helper function for Slate_cluster 
def election_to_simplex_coords(election, slates, num_cands = 'Auto', method = 'borda'):
    """
    Returns the simplex coordinates of the given election with respect to the given slates.
    args:
        election: a list of ballots, e.g. [(1,2,3,4,5,6,7), (1,2,3,4,5,6)]
        slates : a list of sets of candidates, e.g. [{1,2},{3,4,5}, {6,7}]
        method : 'borda', 'strong' or 'weak': passed to ballot_to_simplex_coords
    returns: (coords, score):
        coords = a list of the simplex coordinate vectors for each ballot in the election (with repetition)
        score = a weighted sum of the maximum coordinate for each ballot
    """
    if num_cands == 'Auto':
        num_cands = sum(len(slate) for slate in slates)
    to_return_coords = []
    to_return_score = 0
    for ballot, weight in election.items():
        coords = ballot_to_simplex_coords(ballot, slates, num_cands=num_cands, method=method)
        to_return_score += weight*max(coords)
        for _ in range(weight):
            to_return_coords.append(coords)
    return to_return_coords, to_return_score

# helper function for Slate_cluster
def Find_optimal_slates(election, k, num_cands='Auto', method='borda', verbose=False):
    """
    Finds the optimal slates for a given election and number of clusters k.
    args:
        election : a dictionary mapping ballots (tuples) to counts (ints)
        k : number of clusters
        num_cands : number of candidates, or 'Auto' to infer from the election
        method: one of 'borda', 'strong' or 'weak' -- passed to election_to_simplex_coords
    returns:
        a tuple (slates, score) where slates is a list of sets of candidates, score is the average distance of the ballots to the closest slate
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])

    best_score = 0
    best_slates = None

    for slates in  more_itertools.set_partitions(range(1,num_cands+1), k):
        _, score = election_to_simplex_coords(election, slates, num_cands=num_cands, method=method)
        if score > best_score:
            best_score = score
            best_slates = slates
            if verbose:
                print(f"New best score: {best_score} with slates {best_slates}")

    return best_slates, best_score

def Slate_cluster(election, k=2, slates='agglom', agglom_dist = 'borda_completion', 
                  agglom_link = 'avg', share_ties=True, return_slates = False):
    """
    Returns a k-clustering using a slate-based method. 

    Args:
        election : dictionary matching ballots to weights.
        k: (int) the number of clusters desired.
        slates: a list of sets of candidates, e.g. [{1,2},{3,4,5},{6,7}].
            or set slates = 'agglom' to automatically generate slates via agglomerative clustering.
            or set slates = 'optimize' to automatically generate slates via optimization.
        agglom_dist: one of {'borda' (d_1), 'borda_completion' (d_2)}: only used if method=='agglom' -- passed to Group_candidates.
        agglom_link: one of {'min', 'avg', 'max'} for single, averaged or complete linkage clustering: only used if method=='agglom' -- passed to Group_candidates.
        share_ties: (boolean) whether to divide between the clusters the weight of a ballot that's equidistance to slates (otherwise, it is assigned to the lowest index cluster)
        
    Returns:
        (if return_slates == False) clustering
        (if return_slates == True) slate_list, clustering
    """
    num_cands = max([item for ranking in election.keys() for item in ranking])

    if slates == 'agglom':
        groupings = Group_candidates(election, num_cands='Auto', method=agglom_dist, link=agglom_link,
                                     return_all=True, verbose = False)
        slates = groupings[num_cands-k]
    elif slates == 'optimize':
        best_score = 0
        best_slates = None
        for my_slates in  more_itertools.set_partitions(range(1,num_cands+1), k):
            _, score = election_to_simplex_coords(election, my_slates, num_cands=num_cands, method='borda')
            if score > best_score:
                best_score = score
                best_slates = my_slates
        slates = [set(S) for S in best_slates]
    else:
        if len(slates) != k:
            raise Exception(f"slates must be a list of {k} sets of candidates or 'agglom' or 'optimize'.")

    C = blocs_from_slates(election, slates, num_cands=num_cands, shared_ties=share_ties, borda_style='pes')
    
    if return_slates:
        return slates, C
    else:
        return C

def Plot_simplex_kde(simplex_coords, filename=None, dpi=300, discrete=False, bins=20, pad=.05):
    '''
    Shows a density plot for the given 2D or 3D simplex coordinates.
    ARGS:
        simplex_coords: list of coordinates (each a list of 2 or 3 nonnegative numbers that sum to 1)
        filename: if provided, saves the plot to this file.
        dpi: (int) resolution of the saved plot
        discrete: (boolean) whether to use a discrete plot or a continuous KDE plot
        bins: (int) number of bins to use for the histogram (only used if discrete=True and dimension = 2)
    '''
    dim = len(simplex_coords[0])

    if dim == 2:
        X = [x for [x,y] in simplex_coords]

        if discrete:
            # Make bins centered at 0 and 1
            edges = np.linspace(0, 1, bins+1)
            bin_width = edges[1] - edges[0]
            # extend slightly beyond 0 and 1
            edges = np.linspace(-bin_width/2, 1+bin_width/2, bins+1)

            plt.hist(X, bins=edges, density=True, 
                    color="blue", edgecolor="k", align="mid")

            # extend x-limits so bars at 0 and 1 are fully visible
            plt.xlim(-bin_width/2, 1+bin_width/2)
            plt.ylabel("Frequency")
        else:
            sns.kdeplot(X, fill=True, bw_adjust=0.5, clip=(0,1))
            plt.xlim(0, 1)
            plt.ylabel(None)

    elif dim == 3:
        # Convert simplex coordinates to 2D points in an equilateral triangle
        A = np.array([0, 0])
        B = np.array([1, 0])
        C = np.array([0.5, np.sqrt(3)/2])
        coords = []
        for x1, x2, x3 in simplex_coords:
            point = x1 * A + x2 * B + x3 * C
            coords.append(point)
        coords = np.array(coords)

        plt.figure(figsize=(6,6))
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
        plt.plot(triangle[:,0], triangle[:,1], 'k-')
        plt.axis('equal')
        plt.axis('off')
        plt.xlim(-pad, 1 + pad)
        plt.ylim(-pad, np.sqrt(3)/2 + pad)

        if discrete:
            # Count multiplicities of unique simplex coordinates
            counter = Counter(map(tuple, simplex_coords))
            for coord, count in counter.items():
                point = coord[0] * A + coord[1] * B + coord[2] * C
                plt.scatter(point[0], point[1], 
                            s=10*count,   # area ∝ count (radius ∝ sqrt(count))
                            color="blue", alpha=0.6, edgecolors="k")
        else:
            # Continuous KDE heatmap
            x, y = coords[:, 0], coords[:, 1]
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
            # Grid over triangle bounding box
            xi, yi = np.mgrid[0:1:200j, 0:np.sqrt(3)/2:200j]
            grid = np.vstack([xi.ravel(), yi.ravel()])
            zi = kde(grid).reshape(xi.shape)

            # Mask points outside the triangle
            mask = (yi <= -np.sqrt(3)*xi + np.sqrt(3)) & (yi <= np.sqrt(3)*xi)
            zi[~mask] = np.nan

            plt.pcolormesh(xi, yi, zi, shading='auto', cmap='hot', alpha=0.8)

    else:
        raise ValueError("Plot_simplex_kde only supports 2-simplex and 3-simplex coordinates.") 

    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()

