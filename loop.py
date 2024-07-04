import numpy as np 
import pandas as pd
import math
import random
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.cluster import KMeans

from Clustering_Functions import *

results = pd.DataFrame(columns=['filename','num_cands', 'num_voters', 'num_unique_ballots',
                                'avg_ballot_len','ballot_lengths', 'parties', 'method',
                                'block_size', 'silB','silH','centroids_H', 'centroids_B', 
                                'medoids_H','medoids_B'])

for ward in range(2, 18): # need to instead iterate over all 1000+ elections.
    filename = f"Data/edinburgh17-{ward:02}.blt"
    print(filename)

    # compute num_cands, num_unique_ballots, num_voters, avg_ballot_len
    election, cand_names, location = parse(filename)
    num_cands = max([item for ranking in election.keys() for item in ranking])
    all_ballots = [ballot for ballot in election.keys() if election[ballot]>0]
    num_unique_ballots = len(all_ballots)
    candidates = sorted(list(set([item for ranking in all_ballots for item in ranking])))
    num_cands = len(candidates)
    num_voters = sum([election[ballot] for ballot in all_ballots])
    avg_ballot_len = sum([len(ballot)*election[ballot] for ballot in election.keys()])/sum(election.values())
    
    # compute dictionary of ballot lengths
    ballot_lengths = {n:0 for n in range(num_cands+1)}
    for ballot in all_ballots:
        l = len(ballot)
        ballot_lengths[l] +=1   
    
    # commpute dictionary of parties
    parties = dict()
    for count in range(len(cand_names)):
        party = cand_names[count][2]
        parties[count+1] = party

    for method in ['meanBC', 'meansBA', 'meanH', 'medoBC', 'medoBA', 'medoH', 'slate']:
        for count in range(2 if method in ['meanBC', 'meansBA', 'meanH'] else 1):
            if method == 'meanBC':
                C = kmeans(election, proxy = 'Borda', borda_style='bord')
            elif method == 'meanBA':
                C = kmeans(election, proxy = 'Borda', borda_style='full_points')
            elif method == 'meanH':
                C = kmeans(election, proxy = 'HH')
            elif method == 'medoBC':
                C = kmedoids(election, proxy = 'Borda', borda_style = 'bord', verbose = False)
            elif method == 'medoBA':
                C = kmedoids(election, proxy = 'Borda', borda_style = 'full_points', verbose = False)
            elif method == 'medoH':
                C = kmedoids(election, proxy = 'HH', verbose = False)
            else:
                C = Slate_cluster(election, verbose = False)

            # compute block size
            block_size = sum(C[0].values())/sum(election.values())

            # compute silhouetted scores 
            labels = []
            XB = []
            XH = [] # first build list of ballot proxies with repititions
            for ballot, weight in election.items():
                for _ in range(weight):
                    XB.append(Borda_vector(ballot, num_cands=num_cands))
                    XH.append(HH_proxy(ballot,num_cands=num_cands))
                    label = 0 if ballot in C[0].keys() else 1
                    labels.append(label)
            silB = silhouette_score(XB,labels,metric='manhattan')
            silH = silhouette_score(XH,labels,metric='manhattan')
            
            # compute the centroids and medoids

            medoids_B = dict()
            medoids_H = dict()
            centroids_B = dict()
            centroids_H = dict()
            for cn in range(2): # cn = cluster number
                centroids_B[cn], medoids_B[cn] = Centroid_and_Medoid(C[cn], proxy='Borda')
                centroids_H[cn], medoids_H[cn] = Centroid_and_Medoid(C[cn], proxy='HH')  

            # save it all in the next row of the dataframe.
            row_num = results.shape[0]
            results.loc[row_num] = [filename,num_cands, num_voters, num_unique_ballots,
                        avg_ballot_len,ballot_lengths, parties, method,
                        block_size, silB, silH, 
                        centroids_H, centroids_B, medoids_H, medoids_B]
            
    results.to_json('results.json')