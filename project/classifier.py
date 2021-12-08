# Train a resnet18-based classifier on images from the Half-Earth dataset.

import torch
import os

import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torchvision.models import resnet18
from torchmetrics.functional import accuracy as tm_accuracy
from torchmetrics.functional import f1
from fastai.vision.all import *

from utils import make_pred_loader

class Classifier(pl.LightningModule):
    """Simple classifier set up so a pre-trained model can be
    used as the `encoder` with a densely connected head for 
    classification.
    """
    def __init__(self, encoder, input_dim, num_classes, lr=1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=0.25),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, num_classes, bias=False)
        )

        self.lr = lr

    def forward(self, x):
        feats = self.encoder(x)
        out = self.fc(feats)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = tm_accuracy(preds, y)
        top5 = tm_accuracy(logits, y, top_k=5)
        f1_macro = f1(preds, y, average="macro", num_classes=self.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_top5", top5, prog_bar=True)
            self.log(f"{stage}_f1-macro", f1_macro, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def cli_main():
    pl.seed_everything(89)

    #----------
    # args    |
    #----------
    parser = ArgumentParser()
    parser.add_argument("--name", default="classifier", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--input_height", default=32, type=int)
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

    train_labels = labels.loc[labels.set == "train"]
    img_order = [o.name.strip(".jpg") for o in get_image_files(os.path.join(args.data, "train"))]
    train_labels = train_labels.loc[img_order]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=89)
    train_idx, val_idx = splitter.split(train_labels.index.values, train_labels.genus.values).__next__()

    input_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=lambda o: genera[o.name.strip(".jpg")],
        splitter=IndexSplitter(val_idx),
        item_tfms=[Resize(args.input_height * 2)],
        batch_tfms=aug_transforms(size=args.input_height, min_scale=0.75)
    )

    dls = input_block.dataloaders(os.path.join(args.data, "train"), batch_size=args.batch_size, shuffle=True)

    examples, _ = dls.one_batch()
    print(f"created dataloader from {os.path.join(args.data, 'train')}")
    #----------
    # model   |
    #----------
    
    encoder = resnet18(pretrained=True)
    encoder = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())

    model = Classifier(encoder, 512, labels.genus.unique().shape[0])

    print("set up model")
    #----------
    # checks  |
    #----------
    feats = model.encoder(examples[:5].cpu())
    y_hat = model.fc(feats)
    print(f"input size: {examples.shape}")
    print(f"feature size: {feats.shape}")
    print(f"output size: {y_hat.shape}")

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
        get_genus = lambda o: genera[o.name.strip(".jpg")]
        # random test set
        rand_dls = make_pred_loader(
            os.path.join(args.data, "test/rand/"), 
            args.input_height, args.batch_size, 
            yfun=get_genus, vocab=dls.vocab,
            device=device
        )
        # herbarium test set
        herb_dls = make_pred_loader(
            os.path.join(args.data, "test/herb/"), 
            args.input_height, args.batch_size, 
            yfun=get_genus, vocab=dls.vocab,
            device=device)
        # species test set
        spp_dls = make_pred_loader(
            os.path.join(args.data, "test/spp/"), 
            args.input_height, args.batch_size, 
            yfun=get_genus, vocab=dls.vocab,
            device=device
        )
        
        results = [trainer.test(model, test_dls)[0] for test_dls in [rand_dls, herb_dls, spp_dls]]
        results = pd.DataFrame(results).assign(set=["random", "instition", "species"])

        results.to_csv(os.path.join(outdir, "test-results.csv"), index=False)

        
if __name__ == "__main__":
    cli_main()
    
