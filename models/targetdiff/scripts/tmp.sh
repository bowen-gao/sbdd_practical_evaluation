#!/bin/bash

# 获取所有 GPU 的空闲内存并选择空闲内存最多的 GPU
selected_gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | awk '{print $1}' | cat -n | sort -k2,2nr | tail -1 | awk '{print $1-1}')

# 打印选中的 GPU ID
echo $selected_gpu
