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
        centers, _, __, C = Slate_cluster(election, verbose=False, dist = 'strong', return_data=True)
    elif method == 'slate_weak':
        centers, _, __, C = Slate_cluster(election, verbose=False, dist = 'weak', return_data=True)
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
            label = 0 if ballot in C[0].keys() else 1
            labels.append(label)
    sil = silhouette_score(XH, labels, metric="manhattan")
    cal = calinski_harabasz_score(XH, labels)
    dav = davies_bouldin_score(XH,labels)
    return sil, cal, dav

def process_election_file_with_method(method, full_filename):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {full_filename} with {method}", flush=True)
    filename = os.path.basename(full_filename)
    num_cands, election, cand_names, location = csv_parse(full_filename)

    # Compute dictionary of parties
    party_list = party_abrevs(cand_names)
    parties = {i: party_list[i-1] for i in range(1, num_cands + 1)} # convert to dictionary
   
    C, centers = process_election_with_method(method, election)
    sil, cal, dav = (compute_scores(C, election, num_cands))
    block_size = sum(C[0].values()) / sum(election.values())

    results = [
            filename,
            num_cands,
            parties,
            method,
            block_size,
            sil,
            cal,
            dav,
            centers,
            {0:C[0], 1:C[1]},
        ]
    return results


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{(start_time).strftime('%H:%M:%S')}] Start time", flush=True)
    print()
    method_list = ["meanBC", "meanBA", "meanH", "medoBC", "medoBA", "medoH", "slate", "slate_weak"]
    filename_list = glob.glob("scot-elex/**/*.csv")
    to_do_list = [(method, filename) for filename in filename_list for method in method_list ]

    n_jobs = 128

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_election_file_with_method)(method, file) for method, file in to_do_list
    )

    # Flatten the list of lists
    results = [item for sublist in results for item in sublist]

    # Convert results into a DataFrame
    results_df = pd.DataFrame(
        results,
        columns=[
            "filename",
            "num_cands",
            "parties",
            "method",
            "block_size",
            "sil",
            "cal",
            "dav",
            "centers",
            "clustering"
        ]
    )

    results_df.to_pickle("results_2025.pkl")
    print()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] End Time", flush=True)
    print(f"[{datetime.now()-start_time}] Elapsed", flush=True)
