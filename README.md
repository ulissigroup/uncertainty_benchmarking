This reposity houses various experiments where we perform ML regressions using various methods.
Our goal is to down-select the ML methods that give us the best prediction accuracy, [uncertainty calibration, and uncertainty sharpness](https://arxiv.org/abs/1807.00263).

# Content

Each folder in this repository contains Jupyter notebooks that document how we used various ML methods to perform regressions on our data, and the results thereof.
Each of these notebooks use the same set of features and data splits that we created from a couple of databases.

# Data wrangling

The `preprocessing` folder shows how we used various APIs to gather and preprocess our data into features.

## Data sources

Our primary data source is the database we created using [GASpy](https://github.com/ulissigroup/GASpy):  GASdb.
Our secondary data source is [Catalysis-Hub](https://www.catalysis-hub.org), although we did not end up continuing to use this data source for [various reasons](./preprocessing/profiling/profile_cathub_feature_space.ipynb).

## Preprocessing

One way to convert our database of atomic structures into features is to fingerprint them using a method outlined in [this](https://www.nature.com/articles/s41929-018-0142-1) paper.
Thus `fingerprint`s in this repository reference this type of feature.

Another way to featurize atomic structures is to use a [`StructureDataTransformer`](https://github.com/ulissigroup/cgcnn/blob/sklearn_refactor/cgcnn/data.py#L378) for a [Crystal Graph Convolutional Neural Network](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301).
Thus `sdt`s in this repository reference this type of feature.


# Dependencies

Refer to our [list of dependencies](./notes/dependencies.md).

