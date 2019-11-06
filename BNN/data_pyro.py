from __future__ import print_function, division
import os
import csv
import re
import json
import functools
import random
import warnings

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from sklearn.preprocessing import OneHotEncoder
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.base import TransformerMixin
import mongo
import copy

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_i
#     return {'atom_fea':torch.cat(batch_atom_fea, dim=0), 
#             'nbr_fea':torch.cat(batch_nbr_fea, dim=0), 
#             'nbr_fea_idx':torch.cat(batch_nbr_fea_idx, dim=0), 
#             'crystal_atom_idx':crystal_atom_idx}, torch.FloatTensor(batch_target)
    return [torch.cat(batch_atom_fea, dim=0), 
            torch.cat(batch_nbr_fea, dim=0), 
            torch.cat(batch_nbr_fea_idx, dim=0), 
            crystal_atom_idx], torch.FloatTensor(batch_target)

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class StructureData():
    """
    
    RE-COMMENT THIS 
    
    
    Parameters
    ----------
    atoms_list: list of ASE atoms objects
        List of ASE atoms objects (final relaxed geometry)
    atoms_list_initial_config: list of ASE atoms objects
        List of ASE atoms objects (initial unrelaxed geometry)
        This is very important if the model will be used to predict
        the properties of unrelaxed structures
    atom_init_loc: str
        The location of the atom_init.json file that contains atomic properties
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset
    use_voronoi: bool
        Controls whether the original (pair distance) or voronoi
        method from pymatgen is used to determine neighbor lists 
        and distances.
    use_fixed_info: bool
        If True, add whether each atom is fixed by ASE constraints as an atomic feature.
        Hypothesized to improve the fit because there is information in the fixed
        atoms being in the bulk
    use_tag: 
        If true, add the ASE tag as an atomic feature
    use_distance:
        If true, for each atom add a graph distance from the atom to the nearest atom
        on the graph that has a tag of 1 (indicated it is an adsorbate atom in our scheme). 
        This allows atoms near the adsorbate to have a higher influence if the model
        deems it helpful.
    train_geometry: str
        If 'final', use the final relaxed structure for input to the graph
        If 'initial' use the initial unrelaxed structure
        If 'final-adsorbate', 'use the initial relax structure for everything with tag=0,
            but add a fixed-edge feature to adsorbate atoms in the final configuration.
            We did this so that the information from adsorbate movement (ex. on-top to bridge)
            is included in the input space, but the final relaxed bond distance is not included.
            This makes the method transferable to the predictions for unrelaxed structures with
            various adsorbate locations
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    """
    def __init__(self, atoms_list, atoms_list_initial_config, atom_init_loc, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123, use_voronoi=True, use_fixed_info=False, use_tag=False, use_distance=False, train_geometry='final-adsorbate'):
        
        #this copy is very important; otherwise things ran, but there was some sort 
        # of shuffle that was affecting the real list, resulting in weird loss
        # loss functions and poor training
        self.atoms_list = copy.deepcopy(atoms_list)
        self.atoms_list_initial_config = copy.deepcopy(atoms_list_initial_config)
        
        self.atom_init_loc = atom_init_loc
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.use_voronoi = use_voronoi
        
        #Load the atom features and gaussian distribution functions
        assert os.path.exists(self.atom_init_loc), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_loc)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        
        #Store some tags inside the object for later use
        self.use_fixed_info = use_fixed_info
        self.use_tag = use_tag
        self.use_distance = use_distance
        self.train_geometry = train_geometry #could be initial, final, or final-adsorbate?

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        atoms = copy.deepcopy(self.atoms_list[idx])
        crystal = AseAtomsAdaptor.get_structure(atoms)
        atoms_initial_config = copy.deepcopy(self.atoms_list_initial_config[idx])
        crystal_initial_config = AseAtomsAdaptor.get_structure(atoms_initial_config)
        
        #Stack the features from atom_init
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
                              
        # If use_tag=True, then add the tag as an atom feature
        if self.use_tag:
            atom_fea = np.hstack([atom_fea,atoms.get_tags().reshape((-1,1))])
            
        # If use_fixed_info=True, then add whether the atom is fixed by ASE constraint to the features
        if self.use_fixed_info:
            fix_loc, = np.where([type(constraint)==FixAtoms for constraint in atoms.constraints])
            fix_atoms_indices = set(atoms.constraints[fix_loc[0]].get_indices())
            fixed_atoms = np.array([i in fix_atoms_indices for i in range(len(atoms))]).reshape((-1,1))
            atom_fea = np.hstack([atom_fea,fixed_atoms])

        # If use_voronoi, then use the voronoi connectivity from pymatgen to determine neighbors and distances
        if self.use_voronoi:
        
            #Get the connectivity array for the initial and final structure
            VC = VoronoiConnectivity(crystal)
            VC_initial_config = VoronoiConnectivity(crystal_initial_config)
            conn = copy.deepcopy(VC.connectivity_array)
            conn_initial_config = copy.deepcopy(VC_initial_config.connectivity_array)
            
            #Iterate through each atom, find it's neighbors, and add their distances
            all_nbrs = []
            
            # Loop over central atom
            for ii in range(0, conn.shape[0]):
                curnbr = []
                
                #Loop over neighbor atoms
                for jj in range(0, conn.shape[1]):
                
                    #Loop over each possible PBC image for the chosen image
                    for kk in range(0,conn.shape[2]):
                        # Only add as a neighbor if the atom is not the currently selected center one and there is connectivity
                        # to that image
                        if jj is not kk and conn[ii][jj][kk] != 0:
                        
                            #Add the neighbor strength depending on train_geometry base
                            if self.train_geometry =='initial':
                                curnbr.append([ii, conn_initial_config[ii][jj][kk]/np.max(conn_initial_config[ii]), jj])
                            elif self.train_geometry =='final':
                                curnbr.append([ii, conn[ii][jj][kk]/np.max(conn[ii]), jj])
                            elif self.train_geometry == 'final-adsorbate':
                                #In order for this to work, each adsorbate atom should be set to tag==1 in the atoms object
                                if (atoms.get_tags()[ii]==1 or atoms.get_tags()[jj]==1):
                                    if conn[ii][jj][kk]/np.max(conn[ii])>0.3:
                                        curnbr.append([ii, 1.0, jj])
                                    else:
                                        curnbr.append([ii, 0.0, jj])
                                else:
                                    curnbr.append([ii, conn_initial_config[ii][jj][kk]/np.max(conn_initial_config[ii]), jj])
                            else:
                                curnbr.append([ii, conn[ii][jj][kk]/np.max(conn[ii]), jj])
                        else:
                            curnbr.append([ii, 0.0, jj])
                all_nbrs.append(np.array(curnbr))
                
            # If use_distance=True, then add the distance to an adsorbate (tag=1) as a feature
            if self.use_distance:
                atom_fea = np.hstack([atom_fea,distance_to_adsorbate_feature(atoms, VC)])

            #Find the strongest neighbors for each atom
            all_nbrs = np.array(all_nbrs)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1],reverse=True) for nbrs in all_nbrs]
            nbr_fea_idx = np.array([list(map(lambda x: x[2],
                                    nbr[:self.max_num_nbr])) for nbr in all_nbrs])
            nbr_fea = np.array([list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                                for nbr in all_nbrs])
                                
            #expand distance one-hot encoding with GDF
            nbr_fea = self.gdf.expand(nbr_fea)
        else:
            all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
            all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
            nbr_fea_idx, nbr_fea = [], []
            for nbr in all_nbrs:
                if len(nbr) < self.max_num_nbr:
                    nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                       [0] * (self.max_num_nbr - len(nbr)))
                    nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                                   [self.radius + 1.] * (self.max_num_nbr -
                                                         len(nbr)))
                else:
                    nbr_fea_idx.append(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.append(list(map(lambda x: x[1],
                                            nbr[:self.max_num_nbr])))
            nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)

        try:
            nbr_fea = torch.Tensor(nbr_fea)
        except RuntimeError:
            print(nbr_fea)
            
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        atom_fea = torch.Tensor(atom_fea)

        return (atom_fea, nbr_fea, nbr_fea_idx)

class ListDataset():
    def __init__(self, list_in):
        self.list = list_in
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]

class StructureDataTransformer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return
    
    def transform(self,X):
        structure_list = [mongo.make_atoms_from_doc(doc) for doc in X]
        structure_list_orig = [mongo.make_atoms_from_doc(doc['initial_configuration']) for doc in X]

        SD = StructureData(structure_list, structure_list_orig, *self.args, **self.kwargs)
        return SD

    def fit(self,*_):
        return self




class MergeDataset(torch.utils.data.Dataset):
    #Simple custom dataset to combine two datasets 
    # (one for input X, one for label y)
    def __init__(
            self,
            X,
            y,
            length=None,
    ):

        self.X = X
        self.y = copy.deepcopy(y)

        len_X = len(X)
        if y is not None:
            len_y = len(y)
            if len_y != len_X:
                raise ValueError("X and y have inconsistent lengths.")
        self._len = len_X

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        X, y = self.X, self.y
        
        if y is not None:
            yi = copy.deepcopy(y[i])
        else:
            yi = np.nan

        return copy.deepcopy(X[i]), yi

def distance_to_adsorbate_feature(atoms, VC, max_dist = 6):    
    # This function looks at an atoms object and attempts to find
    # the minimum distance from each atom to one of the adsorbate 
    # atoms (marked with tag==1)
    conn = copy.deepcopy(VC.connectivity_array)
    conn = np.max(conn,2)

    for i in range(len(conn)):
        conn[i]=conn[i]/np.max(conn[i])

    #get a binary connectivity matrix
    conn=(conn>0.3)*1
    
    #Everything is connected to itself, so add a matrix with zero on the diagonal 
    # and a large number on the off-diagonal
    ident_connection = np.eye(len(conn))
    ident_connection[ident_connection==0]=max_dist+1
    ident_connection[ident_connection==1]=0

    #For each distance, add an array of atoms that can be connected at that distance
    arrays = [ident_connection]
    for i in range(1,max_dist):
        arrays.append((np.linalg.matrix_power(conn,i)>=1)*i+(np.linalg.matrix_power(conn,i)==0)*(max_dist+1))

    #Find the minimum distance from each atom to every other atom (over possible distances)
    arrays=np.min(arrays,0)

    # Find the minimum distance from one of the adsorbate atoms to the other atoms
    min_distance_to_adsorbate = np.min(arrays[atoms.get_tags()==1],0).reshape((-1,1))
    
    #Make sure all of the one hot distance vectors are encoded to the same length. 
    # Encode, return
    min_distance_to_adsorbate[min_distance_to_adsorbate>=max_dist]=max_dist-1
    OHE = OneHotEncoder(categories=[range(max_dist)]).fit(min_distance_to_adsorbate)
    return OHE.transform(min_distance_to_adsorbate).toarray()