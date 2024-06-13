# python scripts/build_data/inject_feat.py \
#     --src_path /data/prot_frag/train_ligand_pocket/train_1m_ptvdn.lmdb \
#     --src_key feat \
#     --tgt_path /data/prot_frag/train_ligand_pocket/train.small.lmdb \
#     --tgt_key ptvdn

python scripts/build_data/inject_feat.py \
    --src_path /data/prot_frag/train_ligand_pocket/valid_frad.lmdb \
    --src_key feat \
    --tgt_path /data/prot_frag/train_ligand_pocket/valid.lmdb \
    --tgt_key frad
