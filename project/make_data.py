# Split the Half-Earth dataset into training and test sets.

import os
import json
import pandas as pd
import shutil

from argparse import ArgumentParser


def sample_at_most(df, at_most=10, random_state=None):
    df = df.copy()
    
    if df.shape[0] > at_most:
        df = df.sample(n=at_most, random_state=random_state)
        
    return df


def main():
    #----------
    # CLI     |
    #----------
    parser = ArgumentParser()
    parser.add_argument("--datapath", default="data/herbarium-2021-fgvc8/train", type=str)
    parser.add_argument("--outpath", default="data/herbarium-2021-fgvc8-sampled", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--metadata", default="metadata.json", type=str)
    parser.add_argument("--species_sample", default=0.1, type=float)
    parser.add_argument("--rand_sample", default=0.1, type=float)
    parser.add_argument("--max_images", default=100, type=int)

    args = parser.parse_args()

    #----------
    # metadata|
    #----------
    with open(os.path.join(args.datapath, args.metadata), "r") as infile:
        metadata = json.load(infile)

    images = pd.DataFrame(metadata["images"])
    institutions = (
        pd.DataFrame(metadata["institutions"])
          .rename(columns={"id": "institution_id", "name": "institution_name"})
    )
    categories = (
        pd.DataFrame(metadata["categories"])
          .assign(genus=lambda df: df["name"].str.split().str[0])
          .rename(columns={"id": "category_id", "name": "species"})
    )

    annotations = (
        pd.DataFrame(metadata["annotations"])
          .merge(categories, on="category_id")
          .merge(institutions, on="institution_id")
    )

    print("original data:")
    print(f"{annotations.species.unique().shape[0]} unique species")
    print(f"{annotations.genus.unique().shape[0]} unique genera")
    print(f"{annotations.family.unique().shape[0]} unique families")
    print(f"{annotations.order.unique().shape[0]} unique orders")
    print(f"{annotations.institution_name.unique().shape[0]} unique herbaria")

    #----------
    # splits  |
    #----------
    kept = annotations.copy()

    print()
    print(f"before splitting: {kept.family.unique().shape[0]} families, {kept.genus.unique().shape[0]} genera, {kept.species.unique().shape[0]} species, {kept.shape[0]} images")

    # cap to max images
    kept = kept.groupby("species").apply(lambda x: sample_at_most(x, at_most=args.max_images, random_state=args.seed))
    print(f"after capping at {args.max_images} images: {kept.family.unique().shape[0]} families, {kept.genus.unique().shape[0]} genera, {kept.species.unique().shape[0]} species, {kept.shape[0]} images")

    # hold out institutions
    held_out_herbs = kept.loc[kept.institution_name != "New York Botanical Garden"]
    kept = kept.loc[kept.institution_name == "New York Botanical Garden"]

    # hold out species
    spp_sample = (
        categories.loc[categories.category_id.isin(kept.category_id)]
            .sample(frac=args.species_sample)
    )

    held_out_spp = kept.loc[kept.category_id.isin(spp_sample.category_id)]
    kept = kept.loc[~kept.category_id.isin(spp_sample.category_id)]

    # hold out random sample
    held_out_rand = kept.sample(frac=args.rand_sample)
    kept = kept.loc[~kept["id"].isin(held_out_rand["id"])]

    # make sure there's more than one specimen for each genus in training data
    kept = kept.loc[kept.genus.isin(kept.genus.value_counts().loc[lambda x: x > 1].index)]

    # remove held out species from unknown genera
    held_out_spp = held_out_spp.loc[held_out_spp.genus.isin(kept.genus)]
    held_out_herbs = held_out_herbs.loc[held_out_herbs.genus.isin(kept.genus)]
    held_out_rand = held_out_rand.loc[held_out_rand.genus.isin(kept.genus)]

    print(f"held out institutions: {held_out_herbs.family.unique().shape[0]} families, {held_out_herbs.genus.unique().shape[0]} genera, {held_out_herbs.species.unique().shape[0]} species, {held_out_herbs.shape[0]} images")
    print(f"held out species: {held_out_spp.family.unique().shape[0]} families, {held_out_spp.genus.unique().shape[0]} genera, {held_out_spp.species.unique().shape[0]} species, {held_out_spp.shape[0]} images")
    print(f"held out sample: {held_out_rand.family.unique().shape[0]} families, {held_out_rand.genus.unique().shape[0]} genera, {held_out_rand.species.unique().shape[0]} species, {held_out_rand.shape[0]} images")
    print(f"remaining: {kept.family.unique().shape[0]} families, {kept.genus.unique().shape[0]} genera, {kept.species.unique().shape[0]} species, {kept.shape[0]} images")
    
    (
        kept.assign(set="train")
          .append(held_out_herbs.assign(set="institutions"))
          .append(held_out_spp.assign(set="species"))
          .append(held_out_rand.assign(set="random"))
    ).to_csv("data/herbarium-2021-fgvc8-sampled/sample-metadata.csv", index=False)

    #----------
    # save    |
    #----------
    os.mkdir(args.outpath)

    os.mkdir(os.path.join(args.outpath, "train"))
    train_paths = images.loc[images["id"].isin(kept["id"])].file_name.to_list()
    for fname in train_paths:
        outdirs = fname.split("/")[1:]
        for i in range(len(outdirs[:-1])):
            if not os.path.exists(os.path.join(args.outpath, "train", *outdirs[:i+1])):
                os.mkdir(os.path.join(args.outpath, "train", *outdirs[:i+1]))

        shutil.copy(os.path.join(args.datapath, fname), 
                    os.path.join(args.outpath, "train", *outdirs))

    os.mkdir(os.path.join(args.outpath, "test"))
    os.mkdir(os.path.join(args.outpath, "test", "random"))
    rand_paths = images.loc[images["id"].isin(held_out_rand["id"])].file_name.to_list()
    for fname in rand_paths:
        outdirs = fname.split("/")[1:]
        for i in range(len(outdirs[:-1])):
            if not os.path.exists(os.path.join(args.outpath, "test/random", *outdirs[:i+1])):
                os.mkdir(os.path.join(args.outpath, "test/random", *outdirs[:i+1]))

        shutil.copy(os.path.join(args.datapath, fname), 
                    os.path.join(args.outpath, "test/random", *outdirs))

    os.mkdir(os.path.join(args.outpath, "test", "species"))
    spp_paths = images.loc[images["id"].isin(held_out_spp["id"])].file_name.to_list()
    for fname in spp_paths:
        outdirs = fname.split("/")[1:]
        for i in range(len(outdirs[:-1])):
            if not os.path.exists(os.path.join(args.outpath, "test/species", *outdirs[:i+1])):
                os.mkdir(os.path.join(args.outpath, "test/species", *outdirs[:i+1]))

        shutil.copy(os.path.join(args.datapath, fname), 
                    os.path.join(args.outpath, "test/species", *outdirs))

    os.mkdir(os.path.join(args.outpath, "test", "institutions"))
    herb_paths = images.loc[images["id"].isin(held_out_herbs["id"])].file_name.to_list()
    for fname in herb_paths:
        outdirs = fname.split("/")[1:]
        for i in range(len(outdirs[:-1])):
            if not os.path.exists(os.path.join(args.outpath, "test/institutions", *outdirs[:i+1])):
                os.mkdir(os.path.join(args.outpath, "test/institutions", *outdirs[:i+1]))

        shutil.copy(os.path.join(args.datapath, fname), 
                    os.path.join(args.outpath, "test/institutions", *outdirs))

    print("finished")


if __name__ == "__main__":
    main()