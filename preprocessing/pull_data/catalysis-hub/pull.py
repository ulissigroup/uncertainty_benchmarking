'''
This script will fetch a bunch of adsorption energy data from Catalysis-hub and
then save it as a `cathub.pkl` object.

Credit to Kirsten Winther from SUNCAT for helping with this.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import io
import re
import math
import pickle
import requests
from tqdm import tqdm
from ase.io import read


def get_catalysis_hub_data(adsorbate, gases, chunk_size=50):
    '''
    Fetches and formats data from Catalysis-Hub.

    Args:
        adsorbate   A string of the adsorbate you want to get the adsorption
                    energy of
        gases       A list of strings for all the gas-phase references of the
                    adsorption reaction
        chunk_size  An integer indicating how many data points you want to
                    query at a time
    Returns:
        data    A list of dictionaries
    '''
    data = _query_catalysis_hub(adsorbate, gases, chunk_size)
    _clean_up_catalysis_hub_data(data)
    return data


def _query_catalysis_hub(adsorbate, gases, chunk_size):
    '''
    Catalysis-hub kind of crashes if we ask for too much information at once.
    To get around this, we:
        - Query it once, but only ask for `chunk_size` data points
        - Figure out where it stopped
        - Ask it for the next `chunk_size` data points that come after the
          place it stopped
        - Iterate until we get it all

    Args:
        adsorbate   A string of the adsorbate you want to get the adsorption
                    energy of
        gases       A list of strings for all the gas-phase references of the
                    adsorption reaction
        chunk_size  An integer indicating how many data points you want to
                    query at a time
    Returns:
        data    A list of dictionaries
    '''
    data = []

    # Initialize the query
    site = 'http://api.catalysis-hub.org/graphql'
    formatted_gases = '+'.join([reactant + 'gas' for reactant in gases])
    query = ('{'
             '  reactions(first: %i, products: "%sstar", reactants: "star+%s") {'
             '    totalCount'
             '    pageInfo {'
             '      endCursor'
             '    }'
             '    edges {'
             '      node {'
             '        Equation'
             '        coverages'
             '        reactionEnergy'
             '        dftCode'
             '        dftFunctional'
             '        username'
             '        pubId'
             '        systems {'
             '          Trajdata'
             '        }'
             '      }'
             '    }'
             '  }'
             '}'
             % (chunk_size, adsorbate, formatted_gases))

    # Query catalysis-hub the first time
    response = requests.post(site, {'query': query}).json()
    total_count = response['data']['reactions']['totalCount']
    data.extend([edge['node'] for edge in response['data']['reactions']['edges']])

    # Perform some initialization for the rest of the queries
    query = query.replace('    totalCount', '')  # Don't need this anymore
    pattern = r'\((\w+)\)'  # Search pattern for fixing each query
    cursor_endpoint = response['data']['reactions']['pageInfo']['endCursor']

    # Ask for the rest
    n_batches = math.ceil((total_count - len(data)) / chunk_size)
    for _ in tqdm(range(n_batches), desc='Pulling %s data' % adsorbate, unit='batch'):

        # Change the starting point of the next query to be the ending point of
        # the last one
        repl = ('(first: %i, after: %s, products: "%sstar", reactants: "star+%s")'
                % (chunk_size, cursor_endpoint, adsorbate, formatted_gases))
        query = re.sub(pattern, repl, query)

        # Get the new results and add it to our output
        response = requests.post(site, {'query': query}).json()
        data.extend([edge['node'] for edge in response['data']['reactions']['edges']])
    return data


def _clean_up_catalysis_hub_data(data):
    '''
    Reformat the data to better suit our needs. The reformatting is done
    in-place.

    Arg:
        data    The output of `_query_catalysis_hub`
    '''
    # Catalysis-hub labels the adsorption energy as 'reactionEnergy'. We
    # relabel it as 'energy', because that's what GASpy people are used to.
    for datum in data:
        datum['energy'] = datum['reactionEnergy']
        del datum['reactionEnergy']

        # Reformat the 'systems' value from a list of dictionaries to a list of
        # `ase.Atoms` objects
        datum['systems'] = [read(io.StringIO(system['Trajdata']), format='json')
                            for system in datum['systems']]


reactions = {'H': ['H2'],
             'N': ['N2'],
             'C': ['CH4', 'H2'],
             'CO': ['CO'],
             'O': ['H2O'],
             'OH': ['H2O'],
             'OOH': ['H2O']}

all_data = {adsorbate: get_catalysis_hub_data(adsorbate, gases)
            for adsorbate, gases in reactions.items()}

with open('cathub.pkl', 'wb') as file_handle:
    pickle.dump(all_data, file_handle)
