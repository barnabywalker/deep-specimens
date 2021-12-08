# Downstream task to evaluate how well features extracted by models pre-trained on
# the Half-Earth image dataset can be used to identify the genus/family/order of a specimen.
# 
# An L2-penalised logistic regression is trained with balanced weights to identify the chosen
# taxonomic level and is evaluated by 5-fold cross-validation.
#
# Can be run as a slurm array task by providing a `config.json` file listing
# the path to the extracted `features`, path to corresponding `labels`, 
# and `target` taxonomic level to predict (genus, family, or order).

import json
import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SCORING = {
    'acc': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1': 'f1_macro'
}

def main():
    #----------
    # CLI     |
    #----------
    parser = ArgumentParser()
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--features", default=None, type=str)
    file_group.add_argument("--config", default=None, type=str)

    parser.add_argument("--labels", default=None, type=str)
    parser.add_argument("--target", default="genus", type=str)
    parser.add_argument("--task_id", default=0, type=int)
    parser.add_argument("--metadata", default="../data/herbarium-2021-fgvc8-sampled/sample-metadata.csv", type=str)

    args = parser.parse_args()

    if (args.labels is None) and (args.features is not None):
        ValueError("You must provide a file location for the feature labels.")

    if not args.config.endswith(".json"):
        ValueError("If you give a config file, it must be a json.")
    

    task_id = args.task_id
    target = args.target

    if args.features is None:
        with open(args.config, "r") as infile:
            config = json.load(infile)
        
        label_file = config[task_id]["labels"]
        feature_file = config[task_id]["features"]
        target = config[task_id]["target"]

    labels = pd.read_csv(label_file)
    features = np.load(feature_file)

    path_parts = feature_file.split("/")[:-1]
    filename = feature_file.split("/")[-1].split("_")[-1].strip(".npy")
    outpath = os.path.join(*path_parts, f"glm-scores_level-{target}_{filename}.csv")

    metadata = pd.read_csv(args.metadata)

    info = metadata.set_index("image_id")[target].to_dict()
    lookup_info = lambda x: info.get(int(x.split("/")[-1].split(".")[0]), "Unknown")

    X = features
    target = labels.label.apply(lambda x: lookup_info(x)).values
    y_unique, y = np.unique(target, return_inverse=True) 

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=89)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])

    scores = cross_validate(pipe, X, y, scoring=SCORING, cv=cv, n_jobs=5)

    pd.DataFrame(scores).to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
