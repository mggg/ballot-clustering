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
import random
import math
import pandas as pd
from itertools import permutations, combinations, chain

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

# Convert the ballot rows to ints while leaving the candidates as strings
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

def Borda_vector(ballot_or_election,num_cands='Auto', borda_style='bord'):
    """
    Returns the Borda vector of the given ballot or election.  The "Borda vector of an election" means the (weighted) sum of the Borda vectors of its ballots.

    Set borda_style = 'standard' for the standard convention that, in an election with n candidates, awards n-k points to the candidate in position k, and awards zero points for missing candidates.
    
    Set borda_style = 'bord' for  to use the "Conservative" convenction that awards n-k-1 points to the candidate in position k, and awards zero points for missing candidates.
    
    Set borda_style = 'full_points' for  to use the "Averaged" convenction that every ballot awards exactly 1+2+\cdots+n Borda points; this is achieved for a short ballot by dividing the unawarded points equally among the missing candidates.

        
    Args:
        ballot_or_election : a single ballot (tuple) or an election (dictionary matching ballots to weights)
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it, but only if an election is given, since a single ballot isn't enough to determine num_cands.
        borda_style : choice of {'standard', 'bord', 'full_points'}
     
    Returns:
        the Borda vector (np.array) of the given ballot or election.                
    """
    L = 1 if borda_style=='bord' else 0
    # Borda vector of a ballot
    if type(ballot_or_election) == tuple:
        if num_cands=='Auto':
            raise Exception("A single ballot is not enough to determine the number of candidates.")
        ballot = ballot_or_election
        to_return = [0 for _ in range(num_cands)]
        for count in range(len(ballot)):
            candidate = ballot[count]
            to_return[candidate-1] = num_cands-count - L
        if borda_style=='full_points':
            missing_cands = set(range(1,num_cands+1))-set(ballot)
            for candidate in missing_cands:
                to_return[candidate-1] += (len(missing_cands)+1)/2
            
    # Borda vector of an election
    else:
        election = ballot_or_election
        if num_cands == 'Auto':
            num_cands = max([item for ranking in election.keys() for item in ranking])

        to_return = [0 for _ in range(num_cands)]
        for ballot, ballot_weight in election.items():
            for count in range(len(ballot)):
                candidate = ballot[count]
                to_return[candidate-1] += ballot_weight*(num_cands-count-L)
            if borda_style=='full_points':
                missing_cands = set(range(1,num_cands+1))-set(ballot)
                for candidate in missing_cands:
                    to_return[candidate-1] += ballot_weight*(len(missing_cands)+1)/2
    
    return np.array(to_return)

def Borda_dist(CA, CB, num_cands = 'Auto', borda_style='bord', order = 1):
    """
    Returns the L^p distance between the Borda vectors of the given pair of ballots or elections,
        where p is called the order (for example, order=2 is the Euclidean distance).

    Set borda_style = 'standard' for the standard convention that, in an election with n candidates, awards n-k points to the candidate in position k, and awards zero points for missing candidates.
    
    Set borda_style = 'bord' for  to use the "Conservative" convenction that awards n-k-1 points to the candidate in position k, and awards zero points for missing candidates.
    
    Set borda_style = 'full_points' for  to use the "Averaged" convenction that every ballot awards exactly 1+2+\cdots+n Borda points; this is achieved for a short ballot by dividing the unawarded points equally among the missing candidates.
    
    Args:
        CA, CB : a pair of ballots (tuples) or elections (dictionaries matching ballots to weights).
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it,
                    but only if an election is given, since a ballot pair isn't enough to determine num_cands.
        borda_style : choice of {'standard', 'bord', 'full_points'}
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
    return np.linalg.norm(VA - VB,ord=order)

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

def Plot_clusters(clusters, method = 'Borda', borda_style='bord', num_cands = 'Auto', order = 'Auto', filename=None, dpi=600):
    """
    Displays a bar plot that helps visualize the given election or clustering.

    Args:
        election: either an election (a dictionary matching ballots to weights) or a clustering (a list of elections).
        method: either 'Borda' for a Borda plot, or 'Mentions' for a stacked mentions plot.
        borda_style: choice of {'bord', 'standard', 'full_points'}, which is passed to Borda_vector.
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        order : Set order='Auto' to order the candidates by deceasing Borda scores in the first cluster.  Set say order=[3,2,4,1] to order the candidates according to the given list. 
        filename : to save the plot.     
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

    palat = ['grey','purple','tomato','orange','b','c','g', 'r', 'm', 'y', 'k']
    r = np.arange(num_cands)
    width = 0.7/k
    bottoms = [np.zeros(num_cands) for _ in range(k)]
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

    if method == 'Borda':
        ax.set_title('Borda Scores of Candidates by Cluster')
    else:
        ax.set_title('Candidate Mentions Stacked by Ballot Position')
    ax.set_xlabel('Candidate')
    plt.xticks(r + (width*(k-1))/2,Ordered_candidates)
    plt.legend()
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi=dpi)

def HH_proxy(ballot,num_cands):
    """
    Returns the head-to-head proxy vector of the given ballot.
        
    This is a vector with one entry for each pair of candidates ordered in the natural way; namely {(1,2),(1,3),...,(1,n),(2,3),...}.  The entries lie in {-1/2,0,1/2} depending on whether the lower-indexed candidate {looses, ties, wins} the head-to-head comparison. 

    Args:
        ballot: a single ballot (tuple)
    
    Returns:
        The head-to-head proxy vector (np.array)
    """
    M = np.zeros([num_cands,num_cands])
    for x,y in combinations(ballot,2):
        M[x-1,y-1] = 1/2
        M[y-1,x-1] = -1/2
    for x in ballot:
        for y in set(range(1,num_cands+1)) - set(ballot): # candidates missing from the ballot
            M[x-1,y-1] = 1/2
            M[y-1,x-1] = -1/2
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
    return np.linalg.norm(H1-H2,ord=order)

def Candidate_dist_matrix(election, num_cands = 'Auto', method = 'borda', trunc = None):
    """ 
    Returns a symmetric matrix whose (i,j) entry is one of these measurements of their 'distance':
    method = 'successive': the portion of ballots on which candidates i & j don't appear next to each other.
    method = 'coappearances': the portion of ballots on which candidates i & j don't both appear.
    method = 'borda' : the average diference in borda_avg points that ballots award to candidates i & j
    method = 'mean_borda' : the average over the completions of the ballots of the diference in the borda points awarded to candidates i & j

    Args
        election : dictionary matching ballots to weights
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        method : one of {'successive', 'coappearances'}
        trunc : truncate all ballots at this position before applying the method.
    """
    if num_cands == 'Auto':
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
    elif method == 'borda' or method == 'mean_borda':
        M = np.zeros([num_cands,num_cands])
        for ballot, weight in election.items():
            trunc_ballot = ballot[:trunc]
            num_missing = num_cands - len(ballot)
            v = Borda_vector(ballot, num_cands=num_cands, borda_style='full_points')
            for i in range(num_cands):
                for j in range(num_cands):
                    M[i,j] += np.abs(v[i]-v[j])*weight
                    if method == 'mean_borda' and v[i]==v[j] and i!=j:
                        M[i,j] += (num_missing+1/3)*weight
    else:
        raise Exception("method must be one of {'successive', 'coappearances', 'borda', 'mean_borda'.")

    return M/sum(election.values())

def Candidate_MDS_plot(election, method = 'mean_borda', num_cands = 'Auto', trunc = None, size_markers = True,
                       party_names = None, party_colors = None, filename = None, dpi = 600):
    """
    Prints a multidimensional scaling (MSD) plot of the candidates, labeled by party.
    The "distance" it approximates is one of the following:
    method = 'successive': the portion of ballots on which candidates i & j don't appear next to each other.
    method = 'coappearances': the portion of ballots on which candidates i & j don't both appear.
    method = 'borda' : the average diference in borda_avg points that ballots award to candidates i & j
    method = 'mean_borda' : the average over the completions of the ballots of the diference in the borda points awarded to candidates i & j


    Args
        election : dictionary matching ballots with weights.
        method : choice of {'successive', 'coappearances', 'borda', 'mean_borda'}
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        trunc : truncate all ballots at this position before applying the method.
        size_markers : (boolean) set to True to size markers by number of first place votes.
        party_names : list of strings used as annotation labels.
        party_colors : 'Auto', None, or list of colors.  Only use 'Auto' of party_names is provided.
        filename : set to None if you don't want to save the plot.
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    M = Candidate_dist_matrix(election, num_cands, method = method, trunc = trunc)
    if party_colors == None:
        party_colors = ['blue' for _ in range(num_cands)]
    elif party_colors == 'Auto':
        D = {'SNP':'yellow', 'Lab': 'red', 'Con':'blue','LD':'orange','Gr':'green'}
        party_colors = []
        for party in party_names:
            party_colors.append(D[party] if party in D.keys() else 'black') 
    
    projections = MDS(n_components=2, dissimilarity='precomputed').fit_transform(M)
    X = np.array([p[0] for p in projections])
    Y = np.array([p[1] for p in projections])
    fig, ax = plt.subplots()

    if size_markers:
        s = [0 for _ in range(num_cands)]
        for ballot,weight in election.items():
            s[ballot[0]-1]+=weight
        ax.scatter(X,Y, c = party_colors, s=.5*np.array(s))
    else:
        ax.scatter(X,Y, c = party_colors)
    
    if not party_names == None:
        for count in range(num_cands):
            ax.annotate(f" {count+1}({party_names[count]})", xy=(X[count], Y[count]))
    ax.grid(False)
    ax.axis('off')
    if filename != None:
        plt.savefig(filename, dpi=dpi)
    plt.show()

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

def Group_candidates(election, num_cands = 'Auto', method = 'mean_borda', trunc = None, link = 'avg'):
    """
    Prints the steps of repeatedly grouping candidates via agglomerative clustering
    using Candidate_dist_matrix as the pairwise distances.

    Args
        election : dictionary matching ballots with weights.
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.
        method : one of {'successive', 'coappearances'}
        trunc : truncate all ballots at this position before applying the method.
        link : one of {'min', 'avg', 'max'} for single, averaged or complete linkage clustering.
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    M = Candidate_dist_matrix(election, num_cands, method = method, trunc = trunc)
    L = [{n} for n in range(1,num_cands+1)]
    print(L)
    while len(L)>1:
        best_val = np.infty
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
        print(L)

def kmeans(election, k=2, proxy='Borda', borda_style='bord', n_init=200, return_centroids=False):
    """
    Returns the clustering obtained by applying the k-means algorithm to the proxies of the ballots.

    Args:
        election : dictionary matching ballots with weights.
        k : the number of clusters desired.
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'bord', 'standard', 'full_points'}, which is passed to Borda_vector (only if proxy == 'Borda') 
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
    
def Manhattan_dist(A,B):
    return sum(np.abs(A-B))

def kmedoids(election, k=2, proxy='Borda', borda_style='bord', verbose = False,
             method = 'pam', share_ties = True, return_medoids=False):
    """
    Returns the clustering obtained by applying the k-medoid algorithm to the proxies of the ballots.

    Args:
        election : dictionary matching ballots with weights.
        k : the number of clusters desired.
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'bord', 'standard', 'full_points'}, which is passed to Borda_vector (only if proxy == 'Borda') 
        verbose : set to True if you want it to print the medoids.
        method : choice of {'pam','alternate'}.  The method 'pam' is more accurate, while 'alternate' is faster
        share_ties : set to True if you want the weight of any ballot that's equidistant to mulitple medoids to be shared between the corresponding clusters in the final iteration. This requires overlaid code because sklearn gives ties to the lowest-indexed cluster (which causes repeatability isses).  
        return_medoids : set to True if you want it to also return the medoids of the returned clustering.

    Returns:
        if return_medoids == False: returns a clustering (list of elections).
        if return_medoids == True: returns a tuple (clustering, medoids).
    """
    num_cands = max([item for ranking in election.keys() for item in ranking])

    # create a matrix whose rows are the ballots (repeated as many times as the ballot was cast) 
    # and a dictionary matching each ballot type with its first corresponding row in the matrix
    # and a reverse dictionary to match each row number of the matrix with a ballot
    X = []
    ballot_to_row = dict()
    row_to_ballot = dict()
    counter = 0
    for ballot, weight in election.items():
        ballot_to_row[ballot]=counter
        for _ in range(weight):
            if proxy=='Borda':
                X.append(Borda_vector(ballot, num_cands=num_cands, borda_style=borda_style))
            else:
                X.append(HH_proxy(ballot,num_cands=num_cands))
            row_to_ballot[counter]=ballot
            counter +=1
    
    model = KMedoids(n_clusters=k, metric="manhattan", method = method, init = 'k-medoids++').fit(X)
    labels = model.labels_
    medoids = model.cluster_centers_
    medoid_ballots = [row_to_ballot[index] for index in model.medoid_indices_]

    if verbose:
        print(f"Medoids = {medoid_ballots}.")

    # convert labels into a clustering (list of dictionaries)
    C = [dict() for _ in range(k)]
    if share_ties:
        total_shared_weight = 0
        for ballot, weight in election.items():
            proxy = X[ballot_to_row[ballot]]
            dists = [Manhattan_dist(medoid,proxy) for medoid in medoids]
            clusts = [x for x in range(k) if dists[x]==np.min(dists)] # multi-valued argmin
            if len(clusts)>1:
                total_shared_weight +=weight
            for clust in clusts:
                C[clust][ballot]=weight/len(clusts)
        #if verbose:
        #    print(f"Portion of ballots that tied = {total_shared_weight/sum(election.values())}")

    else:
        for ballot, weight in election.items():
            lab = labels[ballot_to_row[ballot]]
            C[lab][ballot]=weight

    if return_medoids:
        return C, medoid_ballots
    else:
        return C
    
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

def Clustering_closeness(election,C1,C2, num_cands = 'Auto'):
    """
    Returns the closeness of the given two clusterings, which means the portion of the total ballots for which the two partitions differ (with respect to the best matching of one partition's two clusters with the other's two clusters)
    
    Args:
        election : a dictionary matching ballots to weights.
        C1 : a clustering (list of elections) which must have exactly 2 clusters.
        C2 : a clustering (list of elections) which must have exactly 2 clusters.
        num_cands : the number of candidates.  Set to 'Auto' to ask the algorithm to determine it.

    Returns:
        The closeness of the two clusterings, which equals 0 of they are identical and equals about .5 if they are as unrelated as would be a random pair of clusterings.   
    """
    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])
    matchAA = 0
    matchAB = 0
    for ballot in election.keys():
        W1A = C1[0][ballot] if ballot in C1[0].keys() else 0
        W1B = C1[1][ballot] if ballot in C1[1].keys() else 0
        W2A = C2[0][ballot] if ballot in C2[0].keys() else 0
        W2B = C2[1][ballot] if ballot in C2[1].keys() else 0
        matchAA += np.abs(W1A-W2A) 
        matchAB += np.abs(W1A-W2B)
    return min(matchAA,matchAB)/sum(election.values())

def Centroid_and_Medoid(C, num_cands = 'Auto', proxy='Borda', borda_style='bord', metric = 'Manhattan'):
    """ 
    Returns the centroid and medoid of the given election, C, which will typically be a single cluster.
    The returned centroid is a proxy, while the returned medoid is a ballot.
    
    Args:
        C : an election (typically a single cluster of an election)
        choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'bord', 'standard', 'full_points'}, which is passed to Borda_vector (only if proxy == 'Borda') 
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
            X.append(Borda_vector(ballot, num_cands=num_cands))
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

def Ballot_MDS_plot(election, clusters = None, num_cands = 'Auto', proxy='Borda', borda_style='bord', threshold=10, 
                    label_threshold = np.infty, metric = 'Euclidean', party_names=None, filename=None, dpi = 600):
    """
    Displays an MDS (multi-dimensional scaling) plot for the proxies of all of the ballots in the election that received at least the given threshold number of votes.
    If clusters is provided, they are colored by their cluster assignments; otherwise, by party of 1st place vote.
    
    Args:
        election : a dictionary matching ballots to weights.
        clusters : (optional) a clustering (list of elections that partitions the given election.)
        proxy : choice of {'Borda', 'HH'} for Borda or head-to-head proxy vectors.
        borda_style : choice of {'bord', 'standard', 'full_points'}, which is passed to Borda_vector (only used if proxy == 'Borda') 
        threshold : it ignores all ballots that were cast fewer than the threshold number of times.
        label_threshold : it labels all ballots that were cast at least the label_threshold number of times (set label_threshold=np.infty for no labeling)
        metric : choice of {'Euclidean', 'Manhattan'} for the proxy metric that's approximated.
        party_names : if provided, it will color by party of first place vote.
        filename : to save the plot.   
    """

    if num_cands == 'Auto':
        num_cands = max([item for ranking in election.keys() for item in ranking])

    ballots = []
    proxies = []
    weights = []
    colors = []
    cluster_assignments = []

    if clusters == None:
        clusters = [election]

    for cluster_num in range(len(clusters)):
        cluster = clusters[cluster_num]
        start_index = len(proxies)
        for ballot,weight in cluster.items():
            if weight>=threshold:
                if proxy=='Borda':
                    ballot_proxy = Borda_vector(ballot,num_cands=num_cands, borda_style=borda_style)
                else:
                    ballot_proxy = HH_proxy(ballot,num_cands=num_cands)
                ballots.append(ballot)
                proxies.append(ballot_proxy)
                weights.append(weight)
                cluster_assignments.append(cluster_num)
                if party_names != None:
                    D = {'SNP':'yellow', 'Lab': 'red', 'Con':'blue','LD':'orange','Gr':'green'}
                    party = party_names[ballot[0]-1]
                    colors.append(D[party] if party in D.keys() else 'black')

    if metric == 'Euclidean':
        similarities = euclidean_distances(proxies)
    else:
        similarities = manhattan_distances(proxies)

    projections = MDS(n_components=2, dissimilarity='precomputed').fit_transform(similarities)
    X = np.array([p[0] for p in projections])
    Y = np.array([p[1] for p in projections])

    palat = ['grey','purple','tomato','orange','b','c','g', 'r', 'm', 'y']
    if len(clusters)>1:
        colors = [palat[x] for x in cluster_assignments]
    fig, ax = plt.subplots()
    ax.scatter(X,Y, s = weights, c = colors, alpha = .5)
    ax.set_title('MDS Plot')
    ax.grid(False)
    ax.axis('off')
    for count in range(len(proxies)):
        if weights[count]>label_threshold:
            ax.annotate(ballots[count], xy=(X[count], Y[count]))

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi = dpi)

def powerset(iterable): # returns a list of the nontrival non-full subsets of the given iterable
    """
    Helper function for Slate_cluster   
    """
    s = list(iterable)
    l = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    l.pop()    # remove the full set from the end of the list
    l.pop(0)   # remove the empty set from the start of the list
    return l

# The following returns the pair of HH Proxies for two partial orders
# associated to the slate A
def HH_vectors_of_slate(A,num_cands):
    """
    Helper function for Slate_cluster    
    """
    B = tuple(set(range(1,num_cands+1))-set(A)) # the compliment of A
    M = np.zeros([num_cands,num_cands])
    for x in A:
        for y in B:
            M[x-1,y-1] = 1/2
            M[y-1,x-1] = -1/2
    to_return = []
    for x,y in combinations(range(num_cands),2):
        to_return.append(M[x,y])
    return np.array(to_return), (-1)*np.array(to_return)

def Slate_cluster(election, verbose = True, Delta = True, share_ties = True,
                  return_slates = False):
    """
    Returns a clustering with k=2 clusters using a slate-based method based the distance that ballots are from being strongly consistent.
    
    For each slate S={A,B} (each bi-partition of the candidates), the slate's score is computed as the sum (over the weighted ballots in the election) of the ballot's distance to the closest condition: $A>B$ or $B>A$.
    
    Note that a ballot has zero distance iff it is strongly consistent.  The slate with the minimal score is used to partition the ballots into 2 clusters.

    If verbose == True, the slate is printed coded as a tuple that represents the first half of a bipartition of the candidates. For example the slate (1,3,5) codes for the partition {1,3,5},{2,4,6} (with 6 candidates).

    Args:
        election : dictionary matching ballots to weights.
        verbose : boolean. 
        Delta :  set Delta=False to use the simpler (Delta-free) measurement that says the distance from a ballot and a condition is just the distance between their proxies.
        share_ties  : (boolean) whether to divide between the clusters the weight of a ballot that's equidistance A>B and B>A.
        return_slates : (boolean) whether to also return the slates
        
    Returns:
        A clustering (list of elections).
        (or if return_slates == True) slate_dictionary, a clustering
    """
    num_cands = max([item for ranking in election.keys() for item in ranking])
    # create a matrix X whose rows are the HH proxies of the unique ballots
    # and a dictionary matching each ballot type with its corresponding row in the matrix
    # and a reverse dictionary to match each row number of the matrix with a ballot
    X = []
    ballot_to_row = dict()
    row_to_ballot = dict()
    counter = 0
    num_ballots = 0
    for ballot, weight in election.items():
        num_ballots += weight
        ballot_to_row[ballot]=counter
        row_to_ballot[counter]=ballot
        X.append(HH_proxy(ballot,num_cands=num_cands))
        counter +=1
    
    best_score = float('inf')
    best_subset = tuple()
    
    # Determine the best slate
    for A in powerset(range(1,num_cands+1)):
        B = tuple(set(range(1,num_cands+1))-set(A)) # the compliment of A
        A_slate_size = len(A)
        B_slate_size = len(B)
        A_proxy, B_proxy = HH_vectors_of_slate(A,num_cands)
        slate_score = 0
        
        for ballot, weight in election.items(): # compute dist from the ballot to the slate
            ballot_proxy = X[ballot_to_row[ballot]]
            A_size = len(set(A).intersection(set(ballot)))
            B_size = len(set(B).intersection(set(ballot)))
            diag_points =(math.comb(A_size,2) - math.comb(A_slate_size-A_size,2) \
                        + math.comb(B_size,2) - math.comb(B_slate_size-B_size,2))/2
            A_dist = np.linalg.norm(A_proxy-ballot_proxy,ord=1) - diag_points
            B_dist = np.linalg.norm(B_proxy-ballot_proxy,ord=1) - diag_points
            dist = min(A_dist,B_dist)
            slate_score += dist*weight
        if slate_score<best_score:
            best_score = slate_score
            best_subset = A
    if verbose:
        print(f"Slate = {best_subset}.")

    # Form clusters from the best slate
    A = best_subset
    B = tuple(set(range(1,num_cands+1))-set(A)) # the compliment of A
    A_slate_size = len(A)
    B_slate_size = len(B)
    A_proxy, B_proxy = HH_vectors_of_slate(A,num_cands)
    CA = dict()
    CB = dict()
    total_shared_weight = 0
    
    for ballot, weight in election.items():
        ballot_proxy = X[ballot_to_row[ballot]]
        A_size = len(set(A).intersection(set(ballot)))
        B_size = len(set(B).intersection(set(ballot)))
        if Delta:
            diag_points =(math.comb(A_slate_size,2) - math.comb(A_slate_size-A_size,2) \
                        + math.comb(B_slate_size,2) - math.comb(B_slate_size-B_size,2))/2
        else:
            diag_points = 0
        A_dist = np.linalg.norm(A_proxy-ballot_proxy,ord=1) - diag_points
        B_dist = np.linalg.norm(B_proxy-ballot_proxy,ord=1) - diag_points
        if share_ties and A_dist == B_dist:
            CA[ballot]=weight/2
            CB[ballot]=weight/2
            total_shared_weight +=weight
        elif A_dist<B_dist:
            CA[ballot]=weight
        else:
            CB[ballot]=weight
    # if verbose:
    #    print(f"Portion of ballots that tied = {total_shared_weight/sum(election.values())}")
    
    if return_slates:
        slate_dict = {0:A, 1:B}
        return slate_dict, (CA,CB)
    else:
        return CA,CB

