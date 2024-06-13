#############################################################
# this script is used to inject bowenfeat into crossdock dataset
# the bowenfeat is generated from the retrieval augmented pipeline
# the crossdock dataset is saved in /nfs/data/targetdiff_data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final-001.lmdb
# the crossdocked_v1.1_rmsd1.0_pocket10_processed_final-001.lmdb has been backed up to crossdocked_v1.1_rmsd1.0_pocket10_processed_final-001.lmdb.backup


import os
import pickle
import random

import lmdb
import numpy as np
from tqdm import tqdm


def read_lmdb(lmdb_path, mode="idx"):
    """
    Read lmdb file.

    Args:
        lmdb_path (str): Path to the lmdb file.
        mode (str, optional): Read mode. "idx" to follow the idx order, "direct" to read the data directly (use when idx is not continuous).

    Returns:
        list: List of data read from the lmdb file.
    """
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
    data_all = []
    if mode == "idx":
        for idx in tqdm(range(len(keys)), desc="read lmdb {}".format(lmdb_path)):
            ky=f'{idx}'.encode()
            datapoint_pickled = txn.get(ky)
            data_piece = pickle.loads(datapoint_pickled)
            data_all.append(data_piece)
    elif mode == "direct":
        for key in tqdm(keys, desc="read lmdb {}".format(lmdb_path)):
            datapoint_pickled = txn.get(key)
            data_piece = pickle.loads(datapoint_pickled)
            data_all.append((key, data_piece))
    return data_all

def write_lmdb(data, lmdb_path,mode="idx"):
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, max_readers=32, map_size=int(30e9))
    if mode=="idx":
        with env.begin(write=True) as txn:
            for i, d in tqdm(enumerate(data)):
                txn.put(i, pickle.dumps(d))
    elif mode=="direct":
        with env.begin(write=True) as txn:
            for i, d in tqdm(data):
                txn.put(i, pickle.dumps(d))
    env.close()

crossdocked_path = '/nfs/data/targetdiff_data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final-001.lmdb.+chemdiv'
new_crossdocked_path = '/nfs/data/targetdiff_data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final-001.lmdb'

if os.path.exists(new_crossdocked_path):
    os.remove(new_crossdocked_path)
crossdocked_list=read_lmdb(crossdocked_path,mode="direct")

ret_feat_map={}
ret_feat_path_list=["/nfs/data/targetdiff_data/ret_feat_meta/chemdiv_top100_unimol.pkl","/nfs/data/targetdiff_data/ret_feat_meta/chemdiv_top100_unimol_valid.pkl"]
for ret_feat_path in ret_feat_path_list:
    ret_filetype=ret_feat_path.split(".")[-1]
    if ret_filetype=="mdb":
        bowenfeat_list=read_lmdb(ret_feat_path)
        for data_piece in bowenfeat_list:
            pocket_name=data_piece["pocket_name"]
            pocket_name=pocket_name.split(".")[0]
            pocket_name=pocket_name.replace("_pocket10","")
            feat=data_piece["feat"]
            if pocket_name not in ret_feat_map:
                ret_feat_map[pocket_name]=[]
            ret_feat_map[pocket_name].append(feat)
        for key in ret_feat_map.keys():
            assert len(ret_feat_map[key])==10
            ret_feat_map[key]=np.array(ret_feat_map[key])
    elif ret_filetype=="pkl":
        with open(ret_feat_path,"rb") as f:
            ret_feat_data=pickle.load(f)
        data_num=len(ret_feat_data['index'])
        for idx in range(data_num):
            pdb_name=ret_feat_data['index'][idx]
            feat=ret_feat_data['data'][idx]
            key=pdb_name.replace("_pocket10","").replace(".pdb","")
            if key not in ret_feat_map:
                ret_feat_map[key]=[]
            ret_feat_map[key].append(feat)
    else:
        raise NotImplementedError
for key in ret_feat_map.keys():
    assert len(ret_feat_map[key])==100
    ret_feat_map[key]=np.array(ret_feat_map[key]).astype(np.float32)

print("ret_feat_map.keys(): ", list(ret_feat_map.keys())[:10])
new_crossdocked_list=[]
for idx,data_piece in tqdm(crossdocked_list):
    key=data_piece["src_ligand_filename"]
    key=key.split(".")[0]
    assert key in ret_feat_map
    data_piece["ret_feat_unimol"]=ret_feat_map[key]
    new_crossdocked_list.append((idx,data_piece))
write_lmdb(new_crossdocked_list,new_crossdocked_path,mode="direct")
print("done")
