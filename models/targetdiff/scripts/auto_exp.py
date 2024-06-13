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

class AutoExpRunner:
    def __init__(self,args):
        self.exp_name=args.exp_name
        self.train_device=args.train_device
        self.sample_device=args.sample_device
        if self.sample_device is not None:
            self.sample_device=[int(x) for x in self.sample_device.split(",")]
        self.ckpt_path=os.path.join("/data/targetdiff_data/ckpts",self.exp_name+".pt")
        self.sample_output_path=os.path.join("/data/targetdiff_data/PCBA_sample_output",self.exp_name)
        self.sample_parallel_num=1

    def _get_gpu_memory_map(self):
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ]
        )
        gpu_memory = [int(x) for x in result.decode("utf-8").strip().split("\n")]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map

    def _get_best_gpus(self,num_gpus=1,mode="train"):
        gpu_memory_map = self._get_gpu_memory_map()
        sorted_gpu_memory_map = sorted(gpu_memory_map.items(), key=lambda x: x[1], reverse=False)
        ret=[]
        for i in range(num_gpus):
            ret.append(sorted_gpu_memory_map[i][0])
        return ret

    def trained(self):
        if not os.path.exists(self.ckpt_path):
            return False
        ckpt=torch.load(self.ckpt_path,map_location="cpu")
        return ckpt["iteration"]>10000 
    

    def train(self):
        print("#"*50)
        print("Start training")
        print("#"*50)
        #python scripts/train_diffusion.py configs/training.yml --device cuda:2 --wandb with_dc_loss_nll
        gpu_id=self._get_best_gpus()[0]
        if self.train_device is not None:
            gpu_id=self.train_device
            print("Use specified gpu ",gpu_id)
        print("Training with gpu ",gpu_id)
        cmd=f"python scripts/train_diffusion.py configs/training.yml --device cuda:{gpu_id} --wandb "+self.exp_name
        try:
            subprocess.run(cmd, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    def _get_sample_tasks(self):
        all_files=glob.glob(os.path.join(self.sample_output_path,"*.pt"))
        done_list=[int(x.split("/")[-1].split("_")[0]) for x in all_files]
        tasks=list(set(range(15))-set(done_list))
        return tasks
        
    def _sample_worker(self, gpu_id, tasks):
        for task in tasks:
            print("#"*50)
            print(f"Start sampling {task} with gpu {gpu_id}")
            print("#"*50)
            cmd = f"python scripts/sample_diffusion.py configs/sampling.yml --ckpt {self.ckpt_path} --data_id {task} --device cuda:{gpu_id} --result_path {self.sample_output_path}"
            os.system(cmd)

    def sample(self):
        print("#"*50)
        print("Start sampling")
        print("#"*50)

        if not os.path.exists(self.sample_output_path):
            os.makedirs(self.sample_output_path)
        
        while (True):

            tasks = self._get_sample_tasks()
            # gpu_ids = self._get_best_gpus(num_gpus=self.sample_parallel_num,mode="sample")

            if len(tasks)==0:
                print("All tasks have been sampled, sample done")
                break

            task_splits = [tasks[i::self.sample_parallel_num] for i in range(self.sample_parallel_num)]
            print("tasks_splits",task_splits)
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.sample_parallel_num) as executor:
                futures = []
                for i in range(self.sample_parallel_num):
                    if self.sample_device is not None:
                        gpu_id=self.sample_device[i%len(self.sample_device)]
                    else:
                        gpu_id = self._get_best_gpus(num_gpus=1)[0]
                    future = executor.submit(self._sample_worker, gpu_id, task_splits[i])
                    futures.append(future)
                    time.sleep(2)

                concurrent.futures.wait(futures)

    def evaluate(self):
        print("#"*50)
        print("Start evaluating")
        print("#"*50)
        cmd=f"python scripts/evaluate_diffusion.py {self.sample_output_path} --docking_mode vina_score --protein_root /data/lit_pcba/raw"
        if args.save_as_sdf:
            cmd+=" --save_as_sdf"
        os.system(cmd)

    def run(self):
        if self.trained():
            print("Already trained, skip training")
        else:
            self.train()
        self.sample()
        self.evaluate()
        print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--train_device', type=str, default=None)
    parser.add_argument('--sample_device', type=str, default=None)
    parser.add_argument('--save_as_sdf', action='store_true')
    args = parser.parse_args()
    runner=AutoExpRunner(args)
    runner.run()