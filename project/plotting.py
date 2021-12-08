# Useful functions to help plot results.

import matplotlib.pyplot as plt
import numpy as np


def plot_examples(examples, preds=None, n=5, **fig_kwargs):
    """Plot a handful of examples in a grid. If a model is specified,
    predictions will be plotted in a row underneath. If not, the examples
    will just be replicated.
    """

    fig, axes = plt.subplots(ncols=n, nrows=2, **fig_kwargs)
    
    if preds is None:
        preds = examples.cpu().numpy()
    else:
        preds = preds.detach().cpu().numpy()
    
    for i, ax in enumerate(axes[0, :]):
        example = examples[i].cpu().numpy()
        example = np.moveaxis(example, 0, -1)
        ax.imshow(example, cmap="Greys_r")
        ax.set_xticks([])
        ax.set_yticks([])

    for i, ax in enumerate(axes[1, :]):
        example = preds[i]
        example = np.moveaxis(example, 0, -1)
        ax.imshow(example, cmap="Greys_r")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(hspace=0, wspace=0)
    return fig, ax

def plot_embeddings(emb_df, **fig_kwargs):
    fig, ax = plt.subplots(**fig_kwargs)
    emb_df.plot.scatter(x="x1", y="x2", s=1, alpha=0.25, ax=ax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for side, spine in ax.spines.items():
        spine.set_visible(False)

    return fig, ax
    