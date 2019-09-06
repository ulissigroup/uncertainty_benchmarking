import random
import pickle
from gaspy.gasdb import get_adsorption_docs


# Load a selection of documents
docs = get_adsorption_docs('CO', extra_projections={'atoms': '$atoms',
                                                    'results': '$results',
                                                    'calc': '$calc',
                                                    'initial_configuration': '$initial_configuration'})
random.shuffle(docs)

# Save the documents
with open('docs.pkl', 'wb') as file_handle:
    pickle.dump(docs, file_handle)
