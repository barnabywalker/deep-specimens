# Utilities to help other scripts

import umap
import pandas as pd

from fastai.vision.all import *


def make_embeddings(codes, tfm=None, labels=None, **umap_kwargs):
    """Transform extracted features using a UMAP embedding and wrangle
    into a dataframe.
    """
    if tfm is None:
        tfm = umap.UMAP(**umap_kwargs).fit(codes)
        embedding = tfm.embedding_
    else:
        embedding = tfm.transform(codes)

    tfm_df = pd.DataFrame({
        "x1": embedding[:,0],
        "x2": embedding[:,1],
        "label": labels
    })

    return tfm, tfm_df


def make_pred_loader(datapath, input_height=32, batch_size=32, yfun=None, vocab=None, device="cuda"):
    """Very hacky way to get all train/validation examples
    into the `valid` part of a fastai dataloader. Training
    and validation examples are treated differently 
    (train randomly cropped, valid center cropped; 
    train truncated to an integer set of batches, valid does all dataset),
    and we want our predictions to be treated like the validation set.
    """

    if yfun is None:
        yfun = lambda o: str(o)
    
    if vocab is None:
        yblock = CategoryBlock
    else:
        yblock = CategoryBlock(vocab=vocab)

    block = DataBlock(
        blocks=(ImageBlock, yblock),
        get_items=get_image_files,
        get_y=yfun,
        # making everything into the validation set
        splitter=FuncSplitter(lambda x: True),
        item_tfms=[Resize(input_height)]
    )

    # making everything in the same split breaks the dls interface
    items = block.get_items(datapath)
    splits = block.splitter(items)
    splits = (splits[1].copy(), splits[1].copy())
    dsets = Datasets(items, tfms=block._combine_type_tfms(), splits=splits,
                     dl_type=block.dl_type, n_inp=block.n_inp)
    
    kwargs = {**block.dls_kwargs, "batch_size": batch_size, "device": device}
    dls = dsets.dataloaders(path=datapath, after_item=block.item_tfms, after_batch=block.batch_tfms,
                            **kwargs)

    return dls.valid