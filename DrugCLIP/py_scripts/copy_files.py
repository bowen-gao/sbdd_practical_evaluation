from tqdm import tqdm


import os

root_path = "/mnt/nfs-ssd/data/pdbbind_2020/combine_set/"
root_path_cur = "/data/protein/pdbbind_2020/combine_set/"

pdbs = os.listdir(root_path_cur)

for pdb in tqdm(pdbs):
    if len(pdb) != 4:
        continue
    # scp
    source = os.path.join(root_path, pdb, f"{pdb}_pocket6A.pdb")
    source = "gaobowen@10.10.10.8:" + source
    dest = "/data/protein/pdbbind_2020/combine_set/" + pdb + "/" + pdb + "_pocket6A.pdb"
    os.system(f"scp {source} {dest}")