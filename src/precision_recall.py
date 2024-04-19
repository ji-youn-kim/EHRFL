# Code referenced from https://github.com/youngjung/improved-precision-and-recall-metric-pytorch

import os
from argparse import ArgumentParser
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

Manifold = namedtuple("Manifold", ["features", "radii"])
PrecisionAndRecall = namedtuple("PrecisinoAndRecall", ["precision", "recall"])


class IPR:
    def __init__(self, k=3):
        self.manifold_ref = None
        self.k = k

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(subject)
        precision = compute_metric(
            self.manifold_ref, manifold_subject.features, "computing precision..."
        )
        recall = compute_metric(
            manifold_subject, self.manifold_ref.features, "computing recall..."
        )
        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, feats):
        self.manifold_ref = self.compute_manifold(feats)

    def compute_manifold(self, feats):
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        return Manifold(feats, radii)


def compute_pairwise_distances(X, Y=None):
    Y = X if Y is None else Y

    distances = torch.nn.functional.pairwise_distance(
        X.unsqueeze(0), Y.unsqueeze(1), p=2, eps=1e-06, keepdim=False
    )
    return distances.numpy()


def distances2radii(distances, k=3):
    num_samples = distances.shape[0]
    radii = np.zeros(num_samples)
    for i in range(num_samples):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k + 1  
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


def compute_metric(manifold_ref, feats_subject, desc=""):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in range(num_subjects):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects


def load_data(path):
    tensors = []
    for fname in os.listdir(path):
        tensors.append(
            torch.load(os.path.join(path, fname), map_location=torch.device("cpu"))
        )
    tensors = torch.cat(tensors, dim=0)
    print(tensors.shape)
    return tensors


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--subjects", type=str, required=True, help="Subject names in comma separated format")
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.subjects = args.subjects.split(",")

    ipr = IPR(k=args.k)

    results = []

    host_path = os.path.join(args.data_path, host)
    host_data = load_data(os.path.join(host_path, host))
    ipr.compute_manifold_ref(host_data)
    for subject in args.subjects:
        subject_data = load_data(os.path.join(host_path, subject))
        precision, recall = ipr.precision_and_recall(subject_data)
        print(f"[model {args.trained_data_size}, data {args.latent_data_size}] {host} -> {subject}: precision={precision}, recall={recall}")
        results.append(
            {
                "host": host,
                "subject": subject,
                "precision": precision,
                "recall": recall,
            }
        )
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.data_path, f"results.csv"), index=False)


if __name__ == "__main__":
    main()
