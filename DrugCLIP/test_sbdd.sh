results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"
MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="pocket.lmdb" # path to the pocket file
EMB_DIR="./data/emb" # path to the cached mol embedding file

weight_path="/data/protein/save_dir/affinity/2023-05-06_22-08-56/checkpoint_best.pt"

finetune_mol_model="/data/protein/molecule/pretrain/mol_pre_no_h_220816.pt"
finetune_pocket_model="/data/protein/molecule/pretrain/pocket_pre_220816.pt"


encoder="drugclip" # select from [unimol, drugclip]

metric="vs" # select from [sim, vs, score]

test_path="/data/protein/lib-pcba/raw/lit_pcba/" # should be set to dude or pcba path

model="/drug/sbdd_bench/p2m_pcba/"


CUDA_VISIBLE_DEVICES="7" python ./unimol/test_sbdd.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 511 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --log-interval 100 --log-format simple \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --path $weight_path \
       --encoder $encoder \
       --metric $metric \
       --testset $test_path \
       --model $model 

       