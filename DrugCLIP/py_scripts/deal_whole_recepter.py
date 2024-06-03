

# get all the subdirs in the directory /data/protein/DUD-E/raw/all/


import os
import numpy as np

from biopandas.pdb import PandasPdb
import lmdb
import pickle

def write_lmdb(data, lmdb_path):
    #resume

    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    num = 0
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode('ascii'), pickle.dumps(d))
            num += 1


def read_pdb(path):
    pdb_df = PandasPdb().read_pdb(path)

    coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']
    protein = {'pocket_coordinates': np.array(coord), 
               'pocket_atoms': list(atom_type),
               'residue_name': list(residue_name),
               'residue_type': list(residue_type),
               'pocket': path.split('/')[-2]
    }   
    return protein



targets = os.listdir('/data/protein/DUD-E/raw/all/')

for target in targets:
    receptor_path = '/data/protein/DUD-E/raw/all/' + target + '/receptor.pdb'
    protein = read_pdb(receptor_path)
    
    # remove existing lmdb
    os.system('rm -rf ' + receptor_path.replace('receptor.pdb', 'receptor.lmdb'))
    
    # convert to receptor.lmdb
    write_lmdb([protein], receptor_path.replace('receptor.pdb', 'receptor.lmdb'))


