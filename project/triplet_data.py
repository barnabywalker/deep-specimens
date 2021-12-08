# A fastia-style dataloader for the Triplet Network.
#
# Loads a set of images as the anchors and samples
# the same and different classes to generate the positive
# and negative examples for each one.

import torch
import pandas as pd
import numpy as np
from fastai.vision.all import *


class TripletTransform(Transform):
    def __init__(self, files, splits, label_func):
        self.label_func = label_func

        # making a dataframe for speed, surprisingly
        df = (
            pd.DataFrame({
                "path": list(files), 
                "label": list(files.map(label_func))
            }).assign(split=lambda x: x.index.isin(splits[1]).astype(int))
        )

        self.splbl2files = [{name: group.path.to_list() for name, group in df.loc[df.split == i].groupby("label")}
                            for i in range(2)]

        self.labels = [df.loc[df.split == i].label.unique() for i in range(2)]
        self.valid = {f: self._draw(f, split=1) for f in files[splits[1]]}

    def encodes(self, f):
        pf, nf = self.valid.get(f, self._draw(f, split=0))
        a, p, n = PILImage.create(f), PILImage.create(pf), PILImage.create(nf)
        return a, p, n
    
    def _draw(self, f, split=0):
        pos_cls = self.label_func(f)
        neg_cls = random.choice(L(l for l in self.labels[split] if l != pos_cls))
        return random.choice(self.splbl2files[split][pos_cls]), random.choice(self.splbl2files[split][neg_cls])


def make_triplet_loader(datapath, label_func, input_height, splitter=RandomSplitter(seed=89), 
                        batch_size=32, shuffle=False, val_only=False, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    files = get_image_files(datapath)
    splits = splitter(files)
    if val_only and (len(splits[0]) == 0):
        splits = (splits[1], splits[1] + L(list(np.array(splits[0]) + len(splits[0]))))
        files = L([*files, *files])

    tfm = TripletTransform(files, splits, label_func)
    tls = TfmdLists(files, tfm, splits=splits)

    dls = tls.dataloaders(after_item=[Resize(input_height * 2), ToTensor],
                          after_batch=[IntToFloatTensor, *aug_transforms(size=args.input_height, min_scale=0.75)],
                          batch_size=batch_size, shuffle=shuffle,
                          device=device)

    if val_only:
        dls = dls.valid

    return dls