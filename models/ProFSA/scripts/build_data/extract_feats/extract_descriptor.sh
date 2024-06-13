root_dir=/data/prot_frag/moleculenet_all
root_dir=/data/prot_frag/moleculenet_20240305

python scripts/build_data/extract_feats/csv2smi.py \
    --input_dir $root_dir

for file in $root_dir/*.smi; do
    python -m mordred $file -o ${file%.smi}_desc.csv -p 1
done

python scripts/build_data/extract_feats/desc2pkl.py \
    --input_dir $root_dir
