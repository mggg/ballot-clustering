from sklearn.metrics import silhouette_score
from Clustering_Functions import *
import csv

# Convert the ballot rows to ints while leaving the candidates as strings
def convert_row(row):
    return [int(item) if item.isdigit() else item for item in row]


def csv_parse(filename):
    data = []
    with open(filename, "r") as f:
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


def process_election_with_method(method, election):
    if method == "meanBC":
        C = kmeans(election, proxy="Borda", borda_style="bord")
    elif method == "meanBA":
        C = kmeans(election, proxy="Borda", borda_style="full_points")
    elif method == "meanH":
        C = kmeans(election, proxy="HH")
    elif method == "medoBC":
        C = kmedoids(election, proxy="Borda", borda_style="bord", verbose=False)
    elif method == "medoBA":
        C = kmedoids(election, proxy="Borda", borda_style="full_points", verbose=False)
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
