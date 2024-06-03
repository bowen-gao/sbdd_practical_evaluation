# read from /data/protein/local_data/drug_clip_pdb_general/train_no_test/


from lmdb_utils import read_lmdb, write_lmdb
from tqdm import tqdm

data = read_lmdb('/data/protein/local_data/drug_clip_pdb_general/train_no_test/train.lmdb')



# read from dude_exclude/exclude_dude_30.m8


with open('./dude_exclude/exclude_dude_30.m8', 'r') as f:
    exclude = f.readlines()

exclude = [x.strip() for x in exclude]

print(exclude[:10])



new_data = []

new_valid = []

for d in tqdm(data):
    #print(d['pocket'])
    if d['pocket'] not in exclude:
        new_data.append(d)
    else:
        new_valid.append(d)


print(len(new_data))
print(len(new_valid))




write_lmdb(new_data, '/drug/drugclip_plus/exclude_30_new/train.lmdb')

write_lmdb(new_valid, '/drug/drugclip_plus/exclude_30_new/valid.lmdb')

print('done')




