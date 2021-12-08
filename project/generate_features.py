# Use a pre-trained neural network to extract features from a set of images.

import torch
import os
import yaml
import json

import pytorch_lightning as pl
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from torch import nn
from torchvision.models import resnet18
from PIL import ImageFile
from tqdm import tqdm
from fastai.vision.all import *

from autoencoder import AESigmoid
from classifier import Classifier
from tripletnet import TripletNet
from utils import make_pred_loader

torch.multiprocessing.set_sharing_strategy('file_system')

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    parser.add_argument("--data", default="data/herbarium-2021-fgvc8-sampled", type=str)
    parser.add_argument("--subset", default="", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--metadata", default="data/herbarium-2021-fgvc8-sampled/sample-metadata.csv", type=str)
    parser.add_argument("--log_dir", default="lightning_logs/", type=str)
    parser.add_argument("--cpus", default=None, type=int)

    args = parser.parse_args()

    if not args.config.endswith(".json"):
        ValueError("If you give a config file, it must be a json.")
 
    task_id = args.task_id
    version = args.version
    model_name = args.name
    subset = args.subset
    data_path = args.data

    if args.config is not None:
        with open(args.config, "r") as infile:
            config = json.load(infile)
        
        model_name = config[task_id]["name"]
        version = config[task_id]["version"]
        subset = config[task_id]["subset"]
        data_path = config[task_id]["data"]

    #----------
    # dirs    |
    #----------
    
    if version is None:
        versions = os.listdir(os.path.join(args.log_dir, model_name))
        version = int(versions[-1].split("_")[-1])
    
    outname = data_path.split("/")[-1]
    outname = "-".join([outname, subset]).strip("-")
    data_path = os.path.join(data_path, subset)

    log_dir = os.path.join(args.log_dir, model_name, f"version_{version}")
    output_dir = os.path.join("output", model_name, f"version_{version}")

    #----------
    # data    |
    #----------
    device = torch.device("cuda:0" if (args.device == "gpu") else "cpu")

    dls = make_pred_loader(data_path, 256, args.batch_size, device=device)

    n_classes = pd.read_csv(args.metadata).genus.unique().shape[0]

    #-----------
    # model    |
    #-----------
    with open(os.path.join(log_dir, "hparams.yaml"), "r") as infile:
        hparams = yaml.safe_load(infile)

    if "ae" in args.name:
        hparams = {"input_height": hparams['input_height']}
        model = AESigmoid(**hparams)
    elif "clf" in args.name:
        encoder = resnet18(pretrained=True)
        encoder = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())
        hparams = {"encoder": encoder, "input_dim": 512, "num_classes": n_classes}
        model = Classifier(**hparams)
    elif "tpl" in args.name:
        encoder = resnet18(pretrained=True)
        encoder = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())
        hparams = {"encoder": encoder}
        model = TripletNet(**hparams)

    latest_chkpt = os.listdir(os.path.join(log_dir, "checkpoints"))[-1]
    model = model.load_from_checkpoint(os.path.join(log_dir, "checkpoints", latest_chkpt),
                                       strict=False, **hparams)

    model = model.encoder.to(device).eval()

    print(f"Recreated model {args.name} from {latest_chkpt}...")  

    #----------
    # latents |
    #----------
    print(f"saving features from {data_path} to {output_dir}...")

    features = []
    labels = []
    for img, label in tqdm(dls, desc="generating features"):
        features.append(model(img).detach().cpu().numpy())
        labels.extend(dls.vocab[label].items)

    features = np.vstack(features)
    with open(os.path.join(output_dir, f"features_{outname}.npy"), "wb") as outfile:
        np.save(outfile, features)

    pd.DataFrame(labels, columns=["label"]).to_csv(os.path.join(output_dir, f"feature-labels_{outname}.csv"), index=False)

    print("finished!")

if __name__ == "__main__":
    main()
