'''
This script will pull data from catalysis hub and then save it as an ase
database. It turns out that catalysis hub has 53 hydrogen adsorption energies
and 446 CO adsorption energies. That's not enough for us, so we don't bother
saving the ase database.
'''

import os
from tqdm import tqdm
import requests
import json
import io
import ase.db
from ase.io import read


query_string =\
"""
{reactions(first: 50, products: "Hstar", reactants: "star+Hgas") {
 totalCount
  edges {
    node {
      chemicalComposition
      surfaceComposition
      facet
      sites
      coverages
      reactants
      products
      Equation
      reactionEnergy
      activationEnergy
      dftCode
      dftFunctional
      username
      pubId
      systems {
      	id
        Trajdata
        keyValuePairs
      }
    }
  }
}}
"""

root = 'http://api.catalysis-hub.org/graphql'
data = requests.post(root, {'query': query_string})

try:
    data = data.json()['data']
except:
    print('Error')

print('%i total structures' % data['reactions']['totalCount'])
try:
    os.remove('reactions.db')
except FileNotFoundError:
    pass

db = ase.db.connect('reactions.db')
for reaction in tqdm(data['reactions']['edges'], desc='Reactions'):
    reaction_data = reaction['node']
    reaction_energy = reaction_data['reactionEnergy']

    for structure in tqdm(reaction_data['systems'], desc='Structures'):
        atoms = read(io.StringIO(structure['Trajdata']), format='json')
        key_value_pairs = json.loads(structure['keyValuePairs'])

        if len(atoms) > 12:
            key_value_pairs.update(adsorption_energy=reaction_energy)
            db.write(atoms, key_value_pairs=key_value_pairs)
print()
