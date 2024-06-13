python scripts/build_data/gen_smi.py \
    --db_path /data/prot_frag/profsa/train.lmdb \
    --smi_path /data/prot_frag/profsa/train.smi

python scripts/build_data/gen_smi.py \
    --db_path /data/prot_frag/profsa/valid.lmdb \
    --smi_path /data/prot_frag/profsa/valid.smi

python -m mordred /data/prot_frag/profsa/train.smi \
    -o /data/prot_frag/profsa/train_desc.csv \
    -p 1

python -m mordred /data/prot_frag/profsa/valid.smi \
    -o /data/prot_frag/profsa/valid_desc.csv \
    -p 1
