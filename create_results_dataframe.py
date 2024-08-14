# Creates results.df, which contains various measurements of various clustering methods 
# run on all 1000+ Scottish elections

import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from datetime import datetime
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from Clustering_Functions import *

def process_election_with_method(method, election):
    if method == "meanBC":
        C, centers = kmeans(election, proxy="Borda", borda_style="pes", return_centroids=True)
    elif method == "meanBA":
        C, centers = kmeans(election, proxy="Borda", borda_style="avg", return_centroids=True)
    elif method == "meanH":
        C, centers = kmeans(election, proxy="HH", return_centroids=True)
    elif method == "medoBC":
        C, centers = kmedoids(election, proxy="Borda", borda_style="pes", verbose=False, return_medoids=True)
    elif method == "medoBA":
        C, centers = kmedoids(election, proxy="Borda", borda_style="avg", verbose=False, return_medoids=True)
    elif method == "medoH":
        C, centers = kmedoids(election, proxy="HH", verbose=False, return_medoids=True)
    elif method == 'slate':
        centers, C = Slate_cluster(election, verbose=False, return_slates=True)
    elif method == 'modularity2':
        C = Modularity_cluster(election, k=2)
        centers = None
    elif method == 'modularity':
        C = Modularity_cluster(election)
        centers = None
    else:
        raise Exception("unknown method")
    if type(centers)==list:
        centers = {0:centers[0], 1:centers[1]} # convert list to dict.
    return C, centers

def compute_scores(C, election, num_cands):
    labels = []
    XH = []  # first build list of ballot proxies with repititions
    for ballot, weight in election.items():
        for _ in range(weight):
            XH.append(HH_proxy(ballot, num_cands=num_cands))
            label = 0
            for count in range(len(C)):
                if ballot in C[count].keys():
                    label = count
            labels.append(label)
    sil = silhouette_score(XH, labels, metric="manhattan")
    cal = calinski_harabasz_score(XH, labels)
    dav = davies_bouldin_score(XH,labels)
    return sil, cal, dav

def process_election_file(full_filename):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {full_filename}", flush=True)
    filename = os.path.basename(full_filename)
    num_cands, election, cand_names, location = csv_parse(full_filename)
    all_ballots = list(election.keys())
    num_unique_ballots = len(all_ballots)
    num_voters = sum(election.values())
    avg_ballot_len = (
        sum([len(ballot) * election[ballot] for ballot in all_ballots]) / num_voters
    )

    ballot_lengths = {n: 0 for n in range(num_cands + 1)}
    for ballot in all_ballots:
        l = len(ballot)
        ballot_lengths[l] += 1

    # Compute dictionary of parties
    parties = dict()
    for count in range(len(cand_names)):
        party = cand_names[count][2]
        parties[count + 1] = party

    results = []
    Clusterings = dict()
    method_list = ["meanBC", "meanBA", "meanH", "medoBC", "medoBA", "medoH",
                    "slate", "modularity2", "modularity"]
    for method in method_list:
        for trial in range(2):
            C, centers = process_election_with_method(method, election)
            if trial == 1:
                sil, cal, dav = (compute_scores(C, election, num_cands))

            Clusterings[(method,trial)]=C
            
            # build a dictionary storing the closenesses of the clusterings formed by the 9 different methods
            # this dictionary will only be included in the dataframe row for modularity2 for this election.
            if method=='modularity' and trial ==1:
                method_closeness = dict()
                for m1 in method_list[:-1]: # all except modularity
                    for m2 in method_list[:-1]:
                        if m1==m2:
                            method_closeness[(m1,m1)]=Clustering_closeness(election,Clusterings[(m1,0)],Clusterings[(m1,1)])
                        else:
                            method_closeness[(m1,m2)]=Clustering_closeness(election,Clusterings[(m1,0)],Clusterings[(m2,0)])
            else:
                method_closeness = None

            block_size = sum(C[0].values()) / num_voters

            if trial == 1:
                results.append(
                    [
                        filename,
                        num_cands,
                        num_voters,
                        num_unique_ballots,
                        avg_ballot_len,
                        ballot_lengths,
                        parties,
                        method,
                        block_size,
                        sil,
                        cal,
                        dav,
                        centers,
                        {n:C[n] for n in range(len(C))},
                        method_closeness
                    ])
    return results


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{(start_time).strftime('%H:%M:%S')}] Start time", flush=True)
    print()
    filename_list = glob.glob("scot-elex/**/*.csv")

    n_jobs = 8

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_election_file)(file) for file in filename_list
    )

    # Flatten the list of lists
    results = [item for sublist in results for item in sublist]

    # Convert results into a DataFrame
    results_df = pd.DataFrame(
        results,
        columns=[
            "filename",
            "num_cands",
            "num_voters",
            "num_unique_ballots",
            "avg_ballot_len",
            "ballot_lengths",
            "parties",
            "method",
            "block_size",
            "sil",
            "cal",
            "dav",
            "centers",
            "clustering",
            "method_closeness"
        ],
    )

    results_df.to_pickle("results_with_modularity.pkl")
    print()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] End Time", flush=True)
    print(f"[{datetime.now()-start_time}] Elapsed", flush=True)
