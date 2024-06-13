python scripts/build_data/build_grdata/merge_lmdb.py \
    /data/screening/smilesdb/smilesdb_litpcba.lmdb litpcba \
    /data/screening/smilesdb/smilesdb.lmdb litpcba

python scripts/build_data/build_grdata/merge_lmdb.py \
    /data/screening/smilesdb/smilesdb_muv.lmdb muv \
    /data/screening/smilesdb/smilesdb.lmdb muv

python scripts/build_data/build_grdata/merge_lmdb.py \
    /data/screening/smilesdb/smilesdb_pcba.lmdb pcba \
    /data/screening/smilesdb/smilesdb.lmdb pcba
