import os
import pickle
import lmdb
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
    print(keys)
    out_list = []
    pocket_list = []
    count=0
    for idx in tqdm(keys):
        datapoint_pickled = txn.get(idx)
        
        data = pickle.loads(datapoint_pickled)

        out_list.append(data)        


    env.close()
    return out_list 



def write_lmdb(data, lmdb_path):
    #resume
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    num = 0
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode(), pickle.dumps(d))
            num += 1







data = [1,2,3,4,5,6,7,8,9,10,11,12,13]

write_lmdb(data,"tt.lmdb")
new = read_lmdb("tt.lmdb")

print(new)
