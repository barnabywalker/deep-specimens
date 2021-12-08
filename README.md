# Harnessing large-scale herbarium image datasets through representation learning

These are files and code used for the analysis descriped in the paper [Harnessing large-scale herbarium image datasets through representation learning](https://www.frontiersin.org/articles/10.3389/fpls.2021.806407/abstract).

In the paper, we trained three different neural network architectures (an autoencoder, a triplet network, and a classifier) on a sample of the [Half-Earth Challenge](https://www.kaggle.com/c/herbarium-2021-fgvc8) dataset of herbarium specimen images. We then used features extracted by these networks in downstream tasks, to assess their generalisability and, subsequently, the potential for representation learning applied to herbarium specimen images.

## Setup

To run the code, you can clone this repository and set up a `conda` environment:

```
conda env create -f environment.yml
```

You then need to create a data folder and save the Half-Earth dataset into it.

You can by visiting the [challenge page](https://www.kaggle.com/c/herbarium-2021-fgvc8) and entering the competition, to gain access. You can then either download the files from the competition data page, or using the Kaggle API.

The Kaggle API provides a command line interface, that should be installed if you set up your python environment as described above. To use the API, you need to download access credentials from your Kaggle competition page.

You can then download the competition files using:
```
kaggle competitions download herbarium-2021-fgvc8 -p ./data/
```

If you get the error `OSError: Could not find kaggle.json. Make sure it's located in {{PATH}}. Or use the environment method.`, move your `kaggle.json` credential file to the path in the error message. 

Once you've downloaded the competition files, unzip them in your data folder.

## Sampling the dataset

The Half-Earth dataset contains > 2M images from 5 institutions for ~65,000 species.

We sampled the dataset to make it more manageable, as well as to create training and test sets.

To do this sampling, run:
```
python project/make_data.py --max_images=25 --species_sample=0.1 --rand_sample=0.1 --seed=89
```

## Training the networks

Once you've set up the environment and prepared the data, you can train the neural networks.

First, you might want to create a folder for the outputs:
```
mkdir output
```

To train the networks in the same way we did, run:

### Autoencoder
```
python project/autoencoder.py --name=resnet18-ae --max_epochs=25 --gpus=1 --input_height=256 --test
```

### Triplet network
```
python project/autoencoder.py --name=resnet18-tpl --max_epochs=25 --gpus=1 --input_height=256 --test
```

### Classifier
```
python project/classifier.py --name=resnet18-clf --max_epochs=25 --gpus=1 --input_height=256 --test
```

## Extracting features

The trained networks can be used to extract features for a set of images, using the script:

```
python project/generate_features.py --name=resnet-18_ae --version=0 --data=data/herbarium-2021-fgvc8-sampled/test --subset=spp
```

## Evaluating specimen representations

We evaluated the representations each network learned by calculating [silhouette scores](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) as a measure of how separable taxonomics groups were, visualising 2D embeddings of the Half-Earth representations, and visualising the activations of the most variable channel in the feature extraction layers.

### Silhouette scores
To calculate the silhouette scores, run:
```
python project/calculate_silhouettes.py --name=resnet18-ae --version=0 
```

For each of the trained networks.

### 2D embeddings
To visualise the 2D embeddings, we used [UMAP](https://umap-learn.readthedocs.io/en/latest/). You can embed a set of features by running:
```
python project/embed_features.py --name=resnet-18-ae --version=0
```
For each of the trained networks. This will use the training images to fit a UMAP embedding and then transform the training and test datasets.

### Channel activations

We used a Jupyter notebook to view channel activations and prepare a figure with examples that maximise and minimise the outputs of a channel. This notebook is in `project/activations_vis.ipynb`.

## Applications

We evaluated how features extracted by the networks generalised to new tasks with three different applications.

### Taxonomic identification at different scales

This application trained an L2-penalised logistic regression to classify the genus, family, or species of features extracted from herbarium specimen images.

We ran it using:
```
python project/downstream_ident.py --features=output/resnet18-ae/version_0/features_test-spp.npy --labels=output/resnet18-ae/version_0/feature-labels_test-spp.csv --target=genus
```

For each taxonomic level `[genus, family, order]`, for each test set `[spp, herb, rand]`, for each trained network.

### Discrimination of similar and distinct genera

This application trained an L2-penalised logistic regression to classify discrimate against two similar and two distinct genera using specimens from RBG, Kew.

We downloaded the specimen images from [iDigBio](https://www.idigbio.org/portal/search). To do this, we searched iDigBio for all occurrences in our genera of interest that had the institution code "K", had the basis of record "PRESERVED_SPECIMEN", and had an image. We downloaded the multimedia and occurrence files for these searches and used the notebook `notebooks/download-idigbio.ipynb` to download the specimen images.

We ran the application using:
```
python project/downstream_discrim.py --model=resnet18-ae --version=0 --genus1=syzygium --genus2=eugenia

python project/downstream_discrim.py --model=resnet18-ae --version=0 --genus1=syzygium --genus2=dendrobium
```
For each trained network.

### Identification of mislabelled specimens

Our final application used the Kew specimen images from the last application, introduced mislabelled specimens, and then used [cleanlab](https://github.com/cleanlab/cleanlab) to try and identify them.

We ran the application using:
```
python project/downstream_mislabel.py --model=resnet18-ae --version=0 --genus1=syzygium --genus2=eugenia --mislabel_frac=0.1

python project/downstream_mislabel.py --model=resnet18-ae --version=0 --genus1=syzygium --genus2=dendrobium --mislabel_frac=0.1
```
For each trained network.

