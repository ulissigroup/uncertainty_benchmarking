'''
The StructureDataTransformer apparently takes a lot of memory. We address this
by dividing our datasets into smaller chunks, and then letting the transformer
handle one chunk at a time. This script divides the data into chunks.
'''

import pickle
from gaspy.utils import _chunk


# Load the documents
with open('../../pull_data/gaspy/docs.pkl', 'rb') as file_handle:
    docs = pickle.load(file_handle)

# Create an iterator that yields the chunks
chunk_size = 5000
docs_iterator = _chunk(docs, chunk_size)

# Save the chunks
for i, _docs in enumerate(docs_iterator):
    with open('gasdb_%i.pkl' % i, 'wb') as file_handle:
        pickle.dump(_docs, file_handle)
