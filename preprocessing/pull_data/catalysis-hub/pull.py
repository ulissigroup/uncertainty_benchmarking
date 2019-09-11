'''
This script will pull data from catalysis hub and then save it as an ase
database. It turns out that catalysis hub has O(10--100) adsorption energies
per adsorbate. That's not enough for us, so we don't bother saving the ase
database.
'''

import os
from tqdm import tqdm
import requests
import json
import io
import ase.db
from ase.io import read


for ads in ['C', 'O', 'N', 'H', 'S', 'OH', 'NH', 'SH', 'CO']:

    # Make the query for catalysis-hub
    query_string =\
    """
    {reactions(first: 50, products: "%sstar", reactants: "star+%sgas") {
     totalCount
      edges {
        node {
          facet
          sites
          coverages
          reactionEnergy
          dftCode
          dftFunctional
          username
          pubId
          systems {
            Trajdata
            keyValuePairs
          }
        }
      }
    }}
    """ % (ads, ads)

    # Grab the data
    root = 'http://api.catalysis-hub.org/graphql'
    data = requests.post(root, {'query': query_string}).json()['data']
    print('%i total structures for %s ' % (data['reactions']['totalCount'], ads))

    # Delete the current `reactions.db` file so we don't add redundant data
    try:
        os.remove('reactions.db')
    except FileNotFoundError:
        pass
    db = ase.db.connect('reactions.db')

    # Get the energies
    for reaction in tqdm(data['reactions']['edges'], desc='Reactions'):
        reaction_data = reaction['node']
        reaction_energy = reaction_data['reactionEnergy']

        # Get the structures
        for structure in tqdm(reaction_data['systems'], desc='Structures'):
            atoms = read(io.StringIO(structure['Trajdata']), format='json')
            key_value_pairs = json.loads(structure['keyValuePairs'])

            # Filter out adsorbates and other small systems
            if len(atoms) > 12:
                key_value_pairs.update(adsorption_energy=reaction_energy)
                db.write(atoms, key_value_pairs=key_value_pairs)
    print()
