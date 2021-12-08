# Calculate silhouette values for features extracted by a pretrained network.
# Silhouette scores (https://www.sciencedirect.com/science/article/pii/0377042787901257)
# are commonly used in cluster analysis as a measure of how well a clustering has performed.
# They balance the average intra-class distance between points with the distance of a point
# to it's nearest neighbour from another class. We're using it here as a measure of how
# the separable taxonomic groups (genus, family, order) are in the extracted feature space.
#
# This can be run ask an array task by providing a `config.json` file detailing the model 
# names and versions to use.

import os
import json

import pytorch_lightning as pl
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm

from sklearn.metrics import silhouette_samples

lookup_info = lambda o, info: info[o.split("/")[-1].strip(".jpg")]

def main():
    pl.seed_everything(89)
    
    #----------
    # CLI     |
    #----------
    parser = ArgumentParser()
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--name", default=None, type=str)
    file_group.add_argument("--config", default=None, type=str)

    parser.add_argument("--version", default=None, type=int)
    parser.add_argument("--task_id", default=0, type=int)
    parser.add_argument("--log_dir", default="lightning_logs/", type=str)
    parser.add_argument("--metadata", default="data/herbarium-2021-fgvc8-sampled/sample-metadata.csv", type=str)

    args = parser.parse_args()

    if not args.config.endswith(".json"):
        ValueError("If you give a config file, it must be a json.")
    
    task_id = args.task_id
    version = args.version
    model_name = args.name

    if args.config is not None:
        with open(args.config, "r") as infile:
            config = json.load(infile)
        
        model_name = config[task_id]["name"]
        version = config[task_id]["version"]
        
    #----------
    # dirs    |
    #----------
    version = args.version
    if version is None:
        versions = os.listdir(os.path.join(args.log_dir, model_name))
        version = int(versions[-1].split("_")[-1])
    
    root_dir = os.path.join("output", model_name, f"version_{version}")

    #----------
    # data    |
    #----------
    features = np.load(os.path.join(root_dir, "features_train.npy"))
    labels = pd.read_csv(os.path.join(root_dir, "feature-labels_train.csv")).label.values

    metadata = (
        pd.read_csv(args.metadata)
          .assign(image_id=lambda df: df.image_id.astype(str))
          .set_index("image_id")
    )
    
    genera = metadata.genus.to_dict()
    families = metadata.family.to_dict()
    orders = metadata["order"].to_dict()

    #--------------
    # silhouettes |
    #--------------
    silhouettes = {"image_id": [label.split("/")[-1].strip(".jpg") for label in labels]}
    for col, info in tqdm([("genus", genera), ("family", families), ("order", orders)]):
        y = np.array([lookup_info(label, info) for label in labels])
        s = silhouette_samples(features, y)
        silhouettes[col] = s
        silhouettes[f"{col}_name"] = y

    pd.DataFrame(silhouettes).to_csv(os.path.join(root_dir, "silhouette-values.csv"), index=False)


if __name__ == "__main__":
    main()