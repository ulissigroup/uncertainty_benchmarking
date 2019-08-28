'''
It takes awhile to apply the CGCNN's StructureDataTransformer (SDT) to all of
our data. Luckily, it can be parallelized across CPU cores. So I separated out
the SDT work out into this script so that I could perform and save it during an
interactive SLURM job.
'''

import random
import multiprocess as mp
import pickle
from tqdm import tqdm
from gaspy.gasdb import get_adsorption_docs
from cgcnn.data import StructureDataTransformer


# Load a selection of documents
docs = get_adsorption_docs('CO', extra_projections={'atoms': '$atoms',
                                                    'results': '$results',
                                                    'calc': '$calc',
                                                    'initial_configuration': '$initial_configuration'})
random.shuffle(docs)

# Save the documents
with open('docs.pkl', 'wb') as file_handle:
    pickle.dump(docs, file_handle)


# Initialize CGCNN transformer
SDT = StructureDataTransformer(atom_init_loc='atom_init.json',
                               max_num_nbr=12,
                               step=0.2,
                               radius=1,
                               use_tag=False,
                               use_fixed_info=False,
                               use_distance=True)

# Run the transformer
SDT_out = SDT.transform(docs)
with mp.Pool(68) as pool:
    iterator = pool.imap(lambda x: SDT_out[x], range(len(SDT_out)), chunksize=40)
    SDT_list = list(tqdm(iterator, total=len(SDT_out)))

# Save the features
with open('sdt.pkl', 'wb') as file_handle:
    pickle.dump(SDT_list, file_handle)


# Pull out feature dimensions for later use by CGCNN
structures = SDT_list[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len = structures[1].shape[-1]

# Save the feature dimensions
with open('feature_dimensions.pkl', 'wb') as file_handle:
    pickle.dump((orig_atom_fea_len, nbr_fea_len), file_handle)