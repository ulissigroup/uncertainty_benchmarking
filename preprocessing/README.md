Here is where we preprocessed our data for the different regression methods.
First we pull our data from our database with the [GASpy](https://github.com/ulissigroup/GASpy) API.

Then to use [CGCNN](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.9b01428), we need to convert the data to a graph format and then to a matrix format.
This was done with the `create_sdt.py` file, which requires the `atom_init.json` file that we provided as a seed.
The `create_sdt.py` file made:  the `docs.pkl` file of raw data; the `sdt.pkl` file of preprocessed data meant for use by CGCNN; and the `feature_dimensions.pkl` file also meant for use by CGCNN.
All of these files contain all of our adsorption energy data for CO.

To use our GP model, we applied the fingerprinting method outlined in our seminal GASpy [paper](https://www.nature.com/articles/s41929-018-0142-1).
This is done by the `fingerprint_docs.ipynb`, which saves the `fingerprints.pkl` file.

Lastly, we do a train/validate/test split using the `split_data.ipynb` notebook, which creates the `splits.pkl` cache.
We use this cache in our ML experiments.
