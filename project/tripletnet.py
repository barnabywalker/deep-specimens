# Train a resnet18-based triplet network on images from the Half-Earth dataset.

import torch
import os

import pytorch_lightning as pl
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from fastai.vision.all import *

from triplet_data import make_triplet_loader

class TripletNet(pl.LightningModule):
    def __init__(self, encoder, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.loss = nn.TripletMarginLoss()

    def forward(self, x):
        return self.encoder(x)

    def step(self, batch, batch_idx):
        a, p, n = batch
        
        za = self.encoder(a)
        zp = self.encoder(p)
        zn = self.encoder(n)

        loss = self.loss(za, zp, zn)
        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def cli_main():
    pl.seed_everything(89)

    #----------
    # args    |
    #----------
    parser = ArgumentParser()
    parser.add_argument("--name", default="autoencoder", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--input_height", default=32, type=int)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)
    parser.add_argument("--data", default="data/herbarium-2021-fgvc8-sampled", type=str)
    parser.add_argument("--label_data", default="data/herbarium-2021-fgvc8-sampled/sample-metadata.csv", type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    print("parsed cli:")
    print(vars(args))

    #----------
    # setup   |
    #----------
    logger = pl.loggers.CSVLogger("lightning_logs", name=args.name)
    version = logger.version
    
    if not os.path.exists(os.path.join("output", args.name)):
        os.mkdir(os.path.join("output", args.name))

    outdir = os.path.join("output", args.name, f"version_{version}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print(f"set up folders, saving to {outdir}")
    #----------
    # data    |
    #----------
    labels = (
        pd.read_csv(args.label_data)
          .assign(image_id=lambda df: df.image_id.astype(str))
          .set_index("image_id")
    )

    genera = labels.genus.to_dict()
    get_genus = lambda o: genera[o.name.strip(".jpg")]

    train_files = get_image_files(os.path.join(args.data, "train"))
    train_genera = train_files.map(get_genus)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=89)
    train_idx, val_idx = splitter.split(train_files, train_genera).__next__()

    dls = make_triplet_loader(
        datapath=os.path.join(args.data, "train"), 
        label_func=get_genus,
        splitter=IndexSplitter(val_idx),
        input_height=args.input_height,
        batch_size=args.batch_size,
        shuffle=True
    )

    examples = dls.one_batch()
    print(f"created dataloader from {os.path.join(args.data, 'train')}")
    print(f"dataloader provides {len(examples)} items each iteration")
    #----------
    # model   |
    #----------
    encoder = resnet18(pretrained=True)
    encoder = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())

    model = TripletNet(encoder)
    print("set up model")
    #----------
    # checks  |
    #----------
    feats = model.encoder(examples[0][:5].cpu())
    print(f"input size: {examples[0].shape}")
    print(f"feature size: {feats.shape}")

    #-----------
    # training |
    #-----------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, dls.train, dls.valid)

    #----------
    # testing |
    #----------
    if args.test:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # random test set
        rand_dls = make_triplet_loader(
            datapath=os.path.join(args.data, "test/rand/"), 
            label_func=lambda o: genera[o.name.strip(".jpg")],
            splitter=FuncSplitter(lambda x: True),
            input_height=args.input_height, 
            batch_size=args.batch_size, 
            device=device,
            shuffle=False,
            val_only=True
        )
        # herbarium test set
        herb_dls = make_triplet_loader(
            datapath=os.path.join(args.data, "test/herb/"), 
            label_func=lambda o: genera[o.name.strip(".jpg")],
            splitter=FuncSplitter(lambda x: True),
            input_height=args.input_height, 
            batch_size=args.batch_size, 
            device=device,
            shuffle=False,
            val_only=True
        )
        # species test set
        spp_dls = make_triplet_loader(
            datapath=os.path.join(args.data, "test/spp/"), 
            label_func=lambda o: genera[o.name.strip(".jpg")],
            splitter=FuncSplitter(lambda x: True),
            input_height=args.input_height, 
            batch_size=args.batch_size, 
            device=device,
            shuffle=False,
            val_only=True
        )
        
        results = [trainer.test(model, test_dls)[0] for test_dls in [rand_dls, herb_dls, spp_dls]]
        results = pd.DataFrame(results).assign(set=["random", "instition", "species"])

        results.to_csv(os.path.join(outdir, "test-results.csv"), index=False)

if __name__ == "__main__":
    cli_main()