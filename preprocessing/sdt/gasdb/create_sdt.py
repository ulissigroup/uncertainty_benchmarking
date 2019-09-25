'''
It takes awhile to apply the CGCNN's StructureDataTransformer (SDT) to all of
our data. Luckily, it can be parallelized across CPU cores. So I separated out
the SDT work out into this script so that I could perform and save it during an
interactive SLURM job.
'''

import os
import multiprocess as mp
import pickle
from tqdm import tqdm
from cgcnn.data import StructureDataTransformer


# Find all the chunks of data
files = os.listdir()
files = [file_ for file_ in files if ('pkl' in file_ and 'gasdb_' in file_)]
files.sort()

# Load the documents
sdts = []
for file_ in tqdm(files, desc='Transforming chunks'):
    with open(file_, 'rb') as file_handle:
        docs = pickle.load(file_handle)

    # Initialize
    SDT = StructureDataTransformer(atom_init_loc='../atom_init.json',
                                   max_num_nbr=12,
                                   step=0.2,
                                   radius=1,
                                   use_tag=False,
                                   use_fixed_info=False,
                                   use_distance=True)
    SDT_out = SDT.transform(docs)

    # Transform
    with mp.Pool(68) as pool:
        iterator = pool.imap(lambda x: SDT_out[x], range(len(SDT_out)), chunksize=40)
        _sdts = list(tqdm(iterator, total=len(SDT_out),
                          desc='Transforming docs in a chunk'))
        sdts.extend(_sdts)

# Save the features
with open('sdts.pkl', 'wb') as file_handle:
    pickle.dump(sdts, file_handle)

# Pull out feature dimensions for later use by CGCNN
structures = sdts[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

# Save the feature dimensions
with open('feature_dimensions.pkl', 'wb') as file_handle:
    pickle.dump((orig_atom_fea_len, nbr_fea_len), file_handle)
