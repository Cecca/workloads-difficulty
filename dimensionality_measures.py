# adapted from https://github.com/Cecca/role-of-dimensionality/blob/fb94a8b1e52c7f71c9fcc5024b5592d68a8c6aac/additional-scripts/compute-lid.py

import numpy as np
import pandas as pd


def compute_lid(distances, k):
    w = distances[min(len(distances) - 1, k)]
    half_w = 0.5 * w

    distances = distances[:k+1]
    distances = distances[distances > 1e-5]

    small = distances[distances < half_w]
    large = distances[distances >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size

    return -valid / s


def compute_rc(distances, k):
    avg_dist = distances.mean()
    return avg_dist / distances[k]


def compute_expansion(distances, k):
    return distances[2*k] / distances[k]


def compute_metrics(query, dataset, k):
    assert query.shape[0] == dataset.shape[1], "data and query are expected to have the same dimension"

    distances = np.linalg.norm(query - dataset, axis=1)
    np.ndarray.sort(distances)

    lid = compute_lid(distances, k)
    rc = compute_rc(distances, k)
    expansion = compute_expansion(distances, k)

    return lid, rc, expansion


def run_benchmark(k=10, runs=10):
    import os
    import h5py
    import faiss
    import time
    from tqdm import tqdm
    import requests

    output_file = "ans.csv"
    if not os.path.isfile(output_file):
        data_file = ".glove-100-angular.hdf5"
        url = "http://ann-benchmarks.com/glove-100-angular.hdf5"
        if not os.path.isfile(data_file):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(data_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)

        with h5py.File(datafile) as hfp:
            dataset = hfp['train'][:]
            queries = hfp['test'][:]

        index = faiss.IndexHNSWFlat(dataset.shape[1], 16)
        index.add(dataset)

        def recall(query, dataset, actual):
            actual = actual[0]
            k = actual.shape[0]
            distances = np.linalg.norm(query - dataset, axis=1)
            idxs = distances.argsort()
            expected = idxs[:k]
            return np.intersect1d(actual, expected).shape[0] / k

        with open(output_file, "w") as fp:
            print("i,lid,rc,expansion,time,recall", file=fp)
            nqueries = queries.shape[0]
            for i in tqdm(range(nqueries)):
                query = queries[i,:]
                lid, rc, expansion = compute_metrics(query, dataset, k)

                qq = np.array([query]) # just to comply with faiss API
                start = time.time()
                for _ in range(runs):
                    index.search(qq, k)
                end = time.time()
                _, nn = index.search(qq, k)
                rec = recall(query, dataset, nn)

                estimate = (end - start) / runs

                print(f"{i}, {lid}, {rc}, {expansion}, {estimate}, {rec}", file=fp)

    return pd.read_csv(output_file)


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    bench = run_benchmark(k = 100, runs = 5)

    print(bench.drop("i", axis=1).corr())

    plt.figure()
    sns.scatterplot(
        data = bench,
        y    = "recall",
        x    = "lid"
    )
    plt.savefig("scatter-recall-lid.png")

    plt.figure()
    sns.scatterplot(
        data = bench,
        y    = "recall",
        x    = "rc"
    )
    plt.savefig("scatter-recall-rc.png")

    plt.figure()
    sns.scatterplot(
        data = bench,
        y    = "recall",
        x    = "expansion"
    )
    plt.savefig("scatter-recall-expansion.png")

