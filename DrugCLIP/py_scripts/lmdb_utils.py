import os
import pickle
import lmdb
import selfies as sf
from tqdm import tqdm, trange



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
        out_list.append(data)


    env.close()
    return out_list 



def write_lmdb(data, lmdb_path):
    #resume
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for i in range(len(data)):
            txn.put(str(i).encode('ascii'), pickle.dumps(data[i]))
    env.close()
