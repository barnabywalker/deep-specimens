# Downstream task to evaluate how well two genera can be distinguished
# in the extracted feature space of a model trained on the Half-Earth
# image dataset. 
# 
# An L2-penalised logistic regression with balanced weights
# is used to perform the discrimination and is evaluated by 5-fold cross-validation.
#
# Can be run as a slurm array task by providing a `config.json` file listing
# the `model_name`, model `version`, `genus1` name, and `genus2` names to run.

import json

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SCORING = {
    'acc': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

def main():
    parser = ArgumentParser()
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--model", default=None, type=str)
    file_group.add_argument("--config", default=None, type=str)

    parser.add_argument("--version", default=None, type=int)
    parser.add_argument("--genus1", default=None, type=str)
    parser.add_argument("--genus2", default=None, type=str)

    parser.add_argument("--task_id", default=None, type=int)

    args = parser.parse_args()

    if (args.version is None) and (args.model is not None):
        ValueError("You must provide a version for your chosen model.")

    if ((args.genus1 is None) and (args.config is None)) or (args.genus2 is None) and (args.config is None):
        ValueError("You must provide the name of two generate to discriminate between.")

    if args.config is not None:
        if not args.config.endswith(".json"):
            ValueError("If you give a config file, it must be a json.")
    
    if (args.task_id is None) and (args.config is not None):
        task_id = 0
    else:
        task_id = args.task_id

    model_name = args.model
    version = args.version
    genus1 = args.genus1
    genus2 = args.genus2

    if args.model is None:
        with open(args.config, "r") as infile:
            config = json.load(infile)
        
        model_name = config[task_id]["model"]
        version = config[task_id]["version"]
        genus1 = config[task_id]["genus1"]
        genus2 = config[task_id]["genus2"]

    outpath = f"output/{model_name}/version_{version}/discrim-scores_{genus1}-{genus2}.csv"

    features = np.vstack([
        np.load(f"output/{model_name}/version_{version}/features_kew-{genus1}.npy"),
        np.load(f"output/{model_name}/version_{version}/features_kew-{genus2}.npy")
    ])

    labels = pd.concat([
        pd.read_csv(f"output/{model_name}/version_{version}/feature-labels_kew-{genus1}.csv").assign(genus=genus1),
        pd.read_csv(f"output/{model_name}/version_{version}/feature-labels_kew-{genus2}.csv").assign(genus=genus2),
    ], ignore_index=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=89)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2500, class_weight="balanced")),
    ])

    X = features
    y = labels.genus == genus1

    scores = cross_validate(pipe, X, y, scoring=SCORING, cv=cv, n_jobs=5)

    pd.DataFrame(scores).to_csv(outpath, index=False)

if __name__ == "__main__":
    main()
