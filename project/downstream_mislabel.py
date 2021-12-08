# Downstream task to evaluate how well features extracted by models pre-trained on
# the Half-Earth image dataset can be used to identify the mislabelled specimens.
# 
# Synthetic mislabelled data is created by switching the labels of a fraction (default = 10 %)
# of specimens. Predicted probabilities are generated for each specimen on the validation set
# of each fold in a cross validation. Predictions are made using an L2-penalised logistic regression
# trained on the training set of each fold. Likely mislabelled specimens are then flagged
# using cleanlab: https://github.com/cleanlab/cleanlab.
#
# Can be run as a slurm array task by providing a `config.json` file listing
# the `model_name`, model `version`, `genus1` name, and `genus2` names to run.

import json

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from cleanlab.pruning import get_noise_indices
from tqdm import tqdm


def generate_mislabels(labels, frac=0.1):
    """Make synthetic mislabelled specimens by switching the labels
    on a random subset of each group. This assumes there are only
    two possible values for genus.
    """
    genera = labels.genus.unique()
    g1 = labels.loc[labels.genus == genera[0]].copy()
    g2 = labels.loc[labels.genus == genera[1]].copy()

    g1_n_wrong = int(np.round(g1.shape[0] * frac))
    g2_n_wrong = int(np.round(g2.shape[0] * frac))

    g1_wrong = np.hstack((np.ones(g1_n_wrong), np.zeros(g1.shape[0] - g1_n_wrong)))
    g2_wrong = np.hstack((np.ones(g2_n_wrong), np.zeros(g2.shape[0] - g2_n_wrong)))

    np.random.shuffle(g1_wrong)
    np.random.shuffle(g2_wrong)

    g1['mislabelled'] = g1_wrong.astype(bool)
    g2['mislabelled'] = g2_wrong.astype(bool)

    g1['genus'] = g1.mislabelled.apply(lambda x: genera[1] if x else genera[0])
    g2['genus'] = g2.mislabelled.apply(lambda x: genera[0] if x else genera[1])

    mislabels = pd.concat([g1, g2], ignore_index=True)

    return mislabels


def main():
    parser = ArgumentParser()
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--model", default=None, type=str)
    file_group.add_argument("--config", default=None, type=str)

    parser.add_argument("--version", default=None, type=int)
    parser.add_argument("--genus1", default=None, type=str)
    parser.add_argument("--genus2", default=None, type=str)
    parser.add_argument("--mislabel_frac", default=0.1, type=float)

    parser.add_argument("--task_id", default=0, type=int)

    args = parser.parse_args()

    if (args.version is None) and (args.model is not None):
        ValueError("You must provide a version for your chosen model.")

    if ((args.genus1 is None) and (args.config is None)) or (args.genus2 is None) and (args.config is None):
        ValueError("You must provide the name of two generate to discriminate between.")

    if args.config is not None:
        if not args.config.endswith(".json"):
            ValueError("If you give a config file, it must be a json.")
    
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

    outpath = f"output/{model_name}/version_{version}/mislabel-scores_{genus1}-{genus2}.csv"

    features = np.vstack([
        np.load(f"output/{model_name}/version_{version}/features_kew-{genus1}.npy"),
        np.load(f"output/{model_name}/version_{version}/features_kew-{genus2}.npy")
    ])

    labels = pd.concat([
        pd.read_csv(f"output/{model_name}/version_{version}/feature-labels_kew-{genus1}.csv").assign(genus=genus1),
        pd.read_csv(f"output/{model_name}/version_{version}/feature-labels_kew-{genus2}.csv").assign(genus=genus2),
    ], ignore_index=True)

    labels = generate_mislabels(labels)

    cv = KFold(n_splits=5, shuffle=True, random_state=89)

    X = features
    y = labels.genus == genus1
    y_wrong = labels.mislabelled

    genus_probs = []
    genus_preds = []
    cv_idx = []
    for train_idx, test_idx in tqdm(cv.split(X, y), total=cv.n_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2500)),
        ])

        fit = pipe.fit(X_train, y_train)
        genus_probs.append(fit.predict_proba(X_test))
        genus_preds.append(fit.predict(X_test))

        cv_idx.append(test_idx)

    genus_probs = np.vstack(genus_probs)
    genus_preds = np.hstack(genus_preds)
    cv_idx = np.hstack(cv_idx)

    noise_idx = get_noise_indices(y[cv_idx], genus_probs, prune_method="prune_by_noise_rate")

    pred_wrong = np.zeros(y_wrong.shape)
    pred_wrong[noise_idx] = 1
    
    tn, fp, fn, tp = confusion_matrix(y_wrong[cv_idx], pred_wrong).ravel()

    scores = [{"genus1": genus1, "genus2": genus2, "fpr": fp / (fp + tn), "tpr": tp / (tp + fn)}]

    pd.DataFrame(scores).to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
