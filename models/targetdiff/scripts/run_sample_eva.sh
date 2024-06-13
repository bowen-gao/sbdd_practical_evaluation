#!/bin/bash

parallel_count=$1
output_path=$2
ckpt=$3
gpus=$4

function cleanup {
  echo "Caught Ctrl+C, cleaning up..."
  pkill -P $$
  exit 1
}

trap cleanup INT

function check_files {
  count=$(ls $1/*.pt 2> /dev/null | wc -l)
  echo $count
}

while true; do
  for (( i=0; i<$parallel_count; i++ ))
  do
    CUDA_VISIBLE_DEVICES=$(echo $gpus | cut -d, -f$((i+1))) bash scripts/batch_sample_diffusion.sh configs/sampling.yml $output_path $parallel_count $i 0 $ckpt &
  done

  wait

  file_count=$(check_files $output_path)
  if [[ $file_count -eq 100 ]]; then
    break
  fi

  echo "Incomplete file set, restarting the loop..."
done

python scripts/evaluate_diffusion.py $output_path --docking_mode vina_dock
