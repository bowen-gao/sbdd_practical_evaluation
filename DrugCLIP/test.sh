results_path="./test"  # replace to your results path
batch_size=8
weight_path="/drug/save_dir/drugclip_plus/2024-04-10_16-23-32/checkpoint_best.pt"



weight_path="/drug/save_dir/drugclip_plus/2024-04-24_20-56-45/checkpoint_best.pt" #50
weight_path="/drug/save_dir/drugclip_plus/2024-04-25_16-32-12/checkpoint_best.pt" #70 
#weight_path="/drug/save_dir/drugclip_plus/2024-04-24_20-57-51/checkpoint_best.pt" #90

TASK="DUDE" # DUDE or PCBA

CUDA_VISIBLE_DEVICES="0" python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --test-task $TASK \