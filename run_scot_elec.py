import pandas as pd
from scot_helper_funcs import (
    csv_parse,
    process_election_with_method,
    compute_centroids_medoids_silhouette,
)
from Clustering_Functions import Slate_cluster
import glob
import os
from joblib import Parallel, delayed
from datetime import datetime


def process_election_file(full_filename):
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
    for method in ["meanBC", "meanBA", "meanH", "medoBC", "medoBA", "medoH", "slate"]:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {filename} calling {method}",
            flush=True,
        )
        count = 2
        #count = 2 if method in ["meanBC", "meanBA", "meanH"] else 1
        for _ in range(count):
            if method == 'slate':
                slates, C = Slate_cluster(election, verbose=False, return_slates = True)
            else:
                slates = None
                C = process_election_with_method(method, election)

            centroids_B, medoids_B, silB, centroids_H, medoids_H, silH = (
                compute_centroids_medoids_silhouette(C, election, num_cands)
            )

            block_size = sum(C[0].values()) / num_voters

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
                    slates
                ]
            )

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
            "slates"
        ],
    )

    results_df.to_pickle("results_with_slates.pkl")
    print()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] End Time", flush=True)
    print(f"[{datetime.now()-start_time}] Elapsed", flush=True)
