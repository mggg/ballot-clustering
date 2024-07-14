# Creates results.df, which contains various measurements of various clustering methods 
# run on all 1000+ Scottish elections

import pandas as pd
import glob
import os
from joblib import Parallel, delayed
from datetime import datetime
from sklearn.metrics import silhouette_score
from Clustering_Functions import *

def process_election_with_method(method, election):
    if method == "meanBC":
        C = kmeans(election, proxy="Borda", borda_style="pes")
    elif method == "meanBA":
        C = kmeans(election, proxy="Borda", borda_style="avg")
    elif method == "meanH":
        C = kmeans(election, proxy="HH")
    elif method == "medoBC":
        C = kmedoids(election, proxy="Borda", borda_style="pes", verbose=False)
    elif method == "medoBA":
        C = kmedoids(election, proxy="Borda", borda_style="avg", verbose=False)
    elif method == "medoH":
        C = kmedoids(election, proxy="HH", verbose=False)
    elif method == 'slate':
        C = Slate_cluster(election, verbose=False)
    else:
        raise Exception("unknown method")

    return C

def compute_centroids_medoids_silhouette(C, election, num_cands):
    labels = []

    XB = []
    XH = []  # first build list of ballot proxies with repititions
    for ballot, weight in election.items():
        for _ in range(weight):
            XB.append(Borda_vector(ballot, num_cands=num_cands))
            XH.append(HH_proxy(ballot, num_cands=num_cands))
            label = 0 if ballot in C[0].keys() else 1
            labels.append(label)
    silB = silhouette_score(XB, labels, metric="manhattan")
    silH = silhouette_score(XH, labels, metric="manhattan")

    # compute the centroids and medoids

    medoids_B = dict()
    medoids_H = dict()
    centroids_B = dict()
    centroids_H = dict()
    for cn in range(2):  # cn = cluster number
        centroids_B[cn], medoids_B[cn] = Centroid_and_Medoid(C[cn], proxy="Borda")
        centroids_H[cn], medoids_H[cn] = Centroid_and_Medoid(C[cn], proxy="HH")

    return centroids_B, medoids_B, silB, centroids_H, medoids_H, silH

def process_election_file(full_filename):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {full_filename}", flush=True)
    filename = os.path.basename(full_filename)
    num_cands, election, cand_names, location = csv_parse(full_filename)
    all_ballots = list(election.keys())
    num_unique_ballots = len(all_ballots)
    candidates = list(range(1, num_cands + 1))
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
    method_list = ["meanBC", "meanBA", "meanH", "medoBC", "medoBA", "medoH", "slate"]
    for method in method_list:
        for trial in range(2):
            if method == 'slate' and trial == 0:
                slates, C = Slate_cluster(election, verbose=False, return_slates = True)
            elif method == 'slate' and trial ==1:
                slates, C = Slate_cluster(election, verbose=False, return_slates = True, Delta = False)
            else:
                slates = None
                C = process_election_with_method(method, election)
            if trial == 0:
                centroids_B, medoids_B, silB, centroids_H, medoids_H, silH = (
                    compute_centroids_medoids_silhouette(C, election, num_cands)
                )

            Clusterings[(method,trial)]=C
            
            # build a dictionary storing the closenesses of the clusterings formed by the 7 different methods
            # this dictionary will only be included in the last dataframe row for this election (slate trial 1).
            if method=='slate' and trial ==1:
                method_closeness = dict()
                for m1 in method_list:
                    for m2 in method_list:
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
                        silB,
                        silH,
                        centroids_H,
                        centroids_B,
                        medoids_H,
                        medoids_B,
                        slates,
                        {0:C[0], 1:C[1]},
                        method_closeness
                    ])
    return results


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{(start_time).strftime('%H:%M:%S')}] Start time", flush=True)
    print()
    filename_list = glob.glob("scot-elex/**/*.csv")

    n_jobs = 128

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
            "silB",
            "silH",
            "centroids_H",
            "centroids_B",
            "medoids_H",
            "medoids_B",
            "slates",
            "clustering",
            "method_closeness"
        ],
    )

    results_df.to_pickle("results.pkl")
    print()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] End Time", flush=True)
    print(f"[{datetime.now()-start_time}] Elapsed", flush=True)
