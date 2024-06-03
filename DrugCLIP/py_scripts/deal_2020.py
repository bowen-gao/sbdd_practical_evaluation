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
        for i in tqdm(range(len(out_list))):
            #print('{:0>10d}'.format(i), pickle.dumps(out_list[i]))
            lmdb_txn.put(str(i).encode('ascii'), pickle.dumps(out_list[i]))

import pickle



def generate_split_lmdb(split_path, save_path, pdb_data):


    with open(split_path, "rb") as f:
        split = pickle.load(f)




    


    # split based on pdbid


    train = split["train"]
    train = [x.split("_")[0] for x in train]

    print(train[:10])
    valid = split["valid"]
    valid = [x.split("_")[0] for x in valid]


    train_list = []
    valid_list = []

    for d in pdb_data:
        if d["pocket"] in train:
            train_list.append(d)
        elif d["pocket"] in valid:
            valid_list.append(d)

    print(len(train_list))

    print(len(valid_list))

    os.makedirs(save_path, exist_ok=True)

    write_lmdb(train_list, f"{save_path}/train.lmdb")
    write_lmdb(valid_list, f"{save_path}/valid.lmdb")



if __name__ == "__main__":
    
    pdb_2020 = read_lmdb("/data/protein/pdbbind_2020/pdb_2020.lmdb")

    #generate_split_lmdb("/drug/rag/data_splits/PDBBind_filtered_by_DUD_E_blast_16569.pkl", "/drug/drugclip_plus/blast_90", pdb_2020)

    #generate_split_lmdb("/drug/rag/data_splits/PDBBind_filtered_by_DUD_E_blast_12830.pkl", "/drug/drugclip_plus/blast_60", pdb_2020)
    # generate_split_lmdb("/drug/drugclip_plus/splits/PDBBind_filtered_by_DUD_E_BLAST_0.9.pkl", "/drug/drugclip_plus/blast_90", pdb_2020)
    # generate_split_lmdb("/drug/drugclip_plus/splits/PDBBind_filtered_by_DUD_E_BLAST_0.6.pkl", "/drug/drugclip_plus/blast_60", pdb_2020)
    # generate_split_lmdb("/drug/drugclip_plus/splits/PDBBind_filtered_by_DUD_E_BLAST_0.3.pkl", "/drug/drugclip_plus/blast_30", pdb_2020)
    generate_split_lmdb("/drug/drugclip_plus/splits/PDBBind_filtered_by_DUD_E_FLAPP_0.5.pkl", "/drug/drugclip_plus/flapp_50", pdb_2020)
    generate_split_lmdb("/drug/drugclip_plus/splits/PDBBind_filtered_by_DUD_E_FLAPP_0.7.pkl", "/drug/drugclip_plus/flapp_70", pdb_2020)
    generate_split_lmdb("/drug/drugclip_plus/splits/PDBBind_filtered_by_DUD_E_FLAPP_0.9.pkl", "/drug/drugclip_plus/flapp_90", pdb_2020)


