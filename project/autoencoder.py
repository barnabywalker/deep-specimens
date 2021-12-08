# Train a resnet18-based autoencoder on images from the Half-Earth dataset.

import torch
import os

import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd

from argparse import ArgumentParser
from torch import nn
from pl_bolts.models.autoencoders import AE
from fastai.vision.all import *

from plotting import plot_examples
from utils import make_pred_loader


class AESigmoid(AE):
    """Simple subclassing of the AE model from pl-bolts.
    This lets us use their pre-trained weights but with
    sigmoid activation on the image reconstruction layer
    because I think this gives better training behaviour.
    """
    def __init__(self, input_height=256):
        print(input_height)
        super().__init__(input_height=input_height)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        feat = self.encoder(x)
        latent = self.fc(feat)
        x_hat = self.decoder(latent)
        return self.sigmoid(x_hat)
    
    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        x_hat = self.sigmoid(x_hat)

        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss

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
    block = DataBlock(
        blocks=(ImageBlock, TransformBlock),
        get_items=get_image_files,
        get_y=lambda o: str(o),
        splitter=RandomSplitter(seed=89),
        item_tfms=[Resize(args.input_height * 2)],
        batch_tfms=aug_transforms(size=args.input_height, min_scale=0.75)
    )

    dls = block.dataloaders(os.path.join(args.data, "train"), batch_size=args.batch_size, shuffle=True)

    examples, _ = dls.one_batch()
    print(f"created dataloader from {os.path.join(args.data, 'train')}")
    #----------
    # model   |
    #----------
    model = AESigmoid(input_height=args.input_height)
    if args.pretrained:
        url = model.pretrained_urls['cifar10-resnet18']
        model = model.load_from_checkpoint(url, strict=False, input_height=args.input_height)
    print("set up model")
    #----------
    # checks  |
    #----------
    feats = model.encoder(examples[:5].cpu())
    latent = model.fc(feats)
    x_hat = model.decoder(latent)
    print(f"input size: {examples.shape}")
    print(f"feature size: {feats.shape}")
    print(f"latent size: {latent.shape}")
    print(f"output size: {x_hat.shape}")

    fig, ax = plot_examples(examples[:5].cpu(), preds=model.sigmoid(x_hat))
    fig.savefig(os.path.join(outdir, "initial_reconstructions.png"))

    #-----------
    # training |
    #-----------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, dls.train, dls.valid)

    #-----------
    # eval     |
    #-----------
    x_hat = model(examples[:5].cpu())
    fig, ax = plot_examples(examples[:5].cpu(), preds=x_hat)
    fig.savefig(os.path.join(outdir, "final_reconstructions.png"))

    #----------
    # testing |
    #----------
    if args.test:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # random test set
        rand_dls = make_pred_loader(os.path.join(args.data, "test/rand/"), args.input_height, args.batch_size, device=device)
        # herbarium test set
        herb_dls = make_pred_loader(os.path.join(args.data, "test/herb/"), args.input_height, args.batch_size, device=device)
        # species test set
        spp_dls = make_pred_loader(os.path.join(args.data, "test/spp/"), args.input_height, args.batch_size, device=device)
        
        results = [trainer.test(model, test_dls)[0] for test_dls in [rand_dls, herb_dls, spp_dls]]
        results = pd.DataFrame(results).assign(set=["random", "instition", "species"])

        results.to_csv(os.path.join(outdir, "test-results.csv"), index=False)

if __name__ == "__main__":
    cli_main()
    
