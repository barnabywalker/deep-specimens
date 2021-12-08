# Make a 2-dimensional embedding of a set of features.
#
# Uses the nonlinear method UMAP (https://umap-learn.readthedocs.io/en/latest/),
# so the structure of local neighbourhoods is preserved but not absolute distances.

import os

import pytorch_lightning as pl
import pandas as pd
import numpy as np

from argparse import ArgumentParser

from plotting import plot_embeddings
from utils import make_embeddings


def main():
    pl.seed_everything(89)
    
    #----------
    # CLI     |
    #----------
    parser = ArgumentParser()
    parser.add_argument("--name", default="classifier", type=str)
    parser.add_argument("--version", default=None, type=int)
    parser.add_argument("--data", default="data/herbarium-2021-fgvc8-sampled", type=str)
    parser.add_argument("--neighbours", default=15, type=int)
    parser.add_argument("--min_dist", default=0.1, type=float)
    parser.add_argument("--label_data", default="data/herbarium-2021-fgvc8-sampled/sample-metadata.csv", type=str)

    args = parser.parse_args()

    #----------
    # dirs    |
    #----------
    version = args.version
    if version is None:
        versions = os.listdir(os.path.join(args.log_dir, args.name))
        version = int(versions[-1].split("_")[-1])
        
    output_dir = os.path.join("output", args.name, f"version_{version}")


    #-----------
    # training |
    #-----------
    print("embedding training features...")

    label_file = os.path.join(output_dir, "feature-labels_train.csv")
    feature_file = os.path.join(output_dir, "features_train.npy")

    labels = pd.read_csv(label_file)
    features = np.load(feature_file)

    tfm, umap_df = make_embeddings(features, labels=labels,
                                   n_neighbors=args.neighbours, min_dist=args.min_dist, random_state=89)

    umap_df.to_csv(os.path.join(output_dir, "feature-embeddings.csv"), index=False)
    fig, ax = plot_embeddings(umap_df, figsize=(10,8))
    fig.savefig(os.path.join(output_dir, "feature-embeddings.png"), dpi=600)

    #----------
    # tests   |
    #----------
    print("embeding random subset...")

    label_file = os.path.join(output_dir, "feature-labels_test-rand.csv")
    feature_file = os.path.join(output_dir, "features_test-rand.npy")

    labels = pd.read_csv(label_file)
    features = np.load(feature_file)

    _, umap_df = make_embeddings(features, tfm=tfm, labels=labels)
    umap_df.to_csv(os.path.join(output_dir, "feature-embeddings_test-rand.csv"), index=False)

    # new herbaria
    print("embedding new herbaria...")

    label_file = os.path.join(output_dir, "feature-labels_test-herb.csv")
    feature_file = os.path.join(output_dir, "features_test-herb.npy")

    labels = pd.read_csv(label_file)
    features = np.load(feature_file)
    
    _, umap_df = make_embeddings(features, tfm=tfm, labels=labels)
    umap_df.to_csv(os.path.join(output_dir, "feature-embeddings_test-herb.csv"), index=False)

    # new species
    print("embedding new species...")

    label_file = os.path.join(output_dir, "feature-labels_test-spp.csv")
    feature_file = os.path.join(output_dir, "features_test-spp.npy")

    labels = pd.read_csv(label_file)
    features = np.load(feature_file)

    _, umap_df = make_embeddings(features, tfm=tfm, labels=labels)
    umap_df.to_csv(os.path.join(output_dir, "feature-embeddings_test-spp.csv"), index=False)

    print("finished!")

if __name__ == "__main__":
    main()
