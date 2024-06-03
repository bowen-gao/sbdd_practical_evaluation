import os
import pickle
import lmdb
import selfies as sf
from tqdm import tqdm, trange

from rdkit import Chem

from rdkit.Chem import AllChem
import numpy as np
from biopandas.pdb import PandasPdb



def read_pdb(path):
    pdb_df = PandasPdb().read_pdb(path)

    coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']
    protein = {'coord': np.array(coord), 
               'atom_type': list(atom_type),
               'residue_name': list(residue_name),
               'residue_type': list(residue_type)}
    return protein

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )  
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    out_list = []
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        
        data = pickle.loads(datapoint_pickled)
        if len(data["pocket_coordinates"])==0:
            continue
        out_list.append(data)

        


    env.close()
    return out_list 



def write_lmdb(out_list, save_path):
    
    env = lmdb.open(
        save_path,
        subdir=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
        map_size=1099511627776
    )

    with env.begin(write=True) as lmdb_txn:
        for i in range(len(out_list)):
            #print('{:0>10d}'.format(i), pickle.dumps(out_list[i]))
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))




root_path = "/drug/DUD-E/raw/"

targets = os.listdir(root_path)

save_path = "/drug/tmp_lmdb_files/"

for target in tqdm(targets):
    #print(target)
    target_path = os.path.join(root_path, target)
    pdb_path = os.path.join(target_path, "pocket6A.pdb")
    
    protein = read_pdb(pdb_path)

    data = {
        "pocket": target,
        "pocket_atoms": protein['atom_type'],
        "pocket_coordinates": protein['coord']
    }

    os.makedirs(os.path.join(save_path, target), exist_ok=True)

    write_lmdb([data], os.path.join(save_path, target, "pocket6a.lmdb"))

