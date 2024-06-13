# automaticallly run a experiment
# Usage: python auto_exp.py <exp_name>
#
# script will run following steps:
# 1. train for 200000 iterations
# 2. sample 20 cases for test set
# 3. evaluate the sampled cases


import os
import sys
import shutil
import argparse
import subprocess
from typing import Any
import torch
import glob
import concurrent.futures
import time
import random 

class AutoExpRunner:
    def __init__(self,args):
        self.exp_name=args.exp_name
        self.main_ckpt_path=os.path.join("/data/AR_data/ckpts",self.exp_name+".pt")
        self.frontier_ckpt_path=os.path.join("/data/AR_data/frontier_ckpts",self.exp_name+".pt")
        if args.save_dir_basename is not None:
            self.sample_output_path=os.path.join("/data/AR_data/PCBA_sample_output",args.save_dir_basename)
        else:
            self.sample_output_path=os.path.join("/data/AR_data/PCBA_sample_output",self.exp_name)
        self.sample_parallel_num=15
        self.device_list=[int(x) for x in args.devices.split(",")]
        print("device_list",self.device_list)
        
    def _sample_worker(self, gpu_id, tasks):
        for task in tasks:
            print("#"*50)
            print(f"Start sampling {task} with gpu {gpu_id}")
            print("#"*50)
            cmd = f"taskset -c {task} python sample.py --data_id {task} --device cuda:{gpu_id} --outdir {self.sample_output_path} --main_ckpt {self.main_ckpt_path} --frontier_ckpt {self.frontier_ckpt_path}"
            print(cmd)
            os.system(cmd)

    def sample(self):
        print("#"*50)
        print("Start sampling")
        print("#"*50)

        if not os.path.exists(self.sample_output_path):
            os.makedirs(self.sample_output_path)

        tasks = list(range(15))
        unfinished_tasks=[]

        for task in tasks:
            sdf_dir=glob.glob(self.sample_output_path+f"/{task}_*")
            if len(sdf_dir)==0:
                unfinished_tasks.append(task)
            else:
                sdf_dir=sdf_dir[0]
                sdf_files=glob.glob(sdf_dir+"/SDF/*.sdf")
                if len(sdf_files)==0:
                    unfinished_tasks.append(task)
        tasks=unfinished_tasks
        self.sample_parallel_num=min(self.sample_parallel_num,len(tasks))
        print("unfinished_tasks",len(tasks))

        task_splits = [tasks[i::self.sample_parallel_num] for i in range(self.sample_parallel_num)]
        print("tasks_splits",task_splits)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.sample_parallel_num) as executor:
            futures = []
            for i in range(self.sample_parallel_num):
                # gpu_id = self._get_best_gpus(num_gpus=1)[0]
                gpu_id = self.device_list[i%len(self.device_list)]
                future = executor.submit(self._sample_worker, gpu_id, task_splits[i])
                futures.append(future)
                time.sleep(1)

            concurrent.futures.wait(futures)


    def run(self):
        self.sample()
        print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--save_dir_basename', type=str, default=None)
    args = parser.parse_args()
    runner=AutoExpRunner(args)
    runner.run()