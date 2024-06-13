import os
import argparse
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor
import time
import concurrent.futures
import torch

class EvaluaionRunner:
    def __init__(self, args):
        self.exp_prefix="/data/AR_data/sample_output/"
        self.protein_root_prefix="/data/DUD-E/raw"
        self.exp_name = args.exp_name
        self.num_process = args.num_process

    def evaluate_case(self, cases):
        for case in cases:
            print(f"Start evaluating {case}")
            case_name=os.path.basename(case).split("_")[-1]
            protein_root = os.path.join(self.protein_root_prefix, case_name , "receptor.pdb")
            # change dir to /project/Pocket2Mol
            os.chdir("/project/Pocket2Mol")
            cmd = f"python /project/Pocket2Mol/evaluation/evaluate.py {os.path.basename(case)} --result_root {os.path.join(self.exp_prefix, self.exp_name)} --protein_root {protein_root}"
            print(cmd)
            subprocess.run(cmd, shell=True)
            print(f"Finish evaluating {case}")

    def run(self):
        case_list = glob.glob(self.exp_prefix + self.exp_name + "/*")
        case_list.sort()
        case_list=[x for x in case_list if os.path.isdir(x)]
        tasks=case_list
        print(f"Total number of cases: {len(case_list)}")

        task_splits = [tasks[i::self.num_process] for i in range(self.num_process)]
        print("tasks_splits",task_splits)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_process) as executor:
            futures = []
            for i in range(self.num_process):
                future = executor.submit(self.evaluate_case, task_splits[i])
                futures.append(future)
                time.sleep(1)

            concurrent.futures.wait(futures)
        
        print("Finish all cases")
        
        # analyze results
        vina_score_list=[]
        vina_min_list=[]
        for case in case_list:
            try:
                result=torch.load(os.path.join(case, "results.pt"))
            except:
                print(f"Error: {case}")
                continue
            for item in result:
                vina_score_list.append(item['vina_score'][0]['affinity'])
                vina_min_list.append(item['vina_min'][0]['affinity'])

        print(f"Vina score mean: {sum(vina_score_list)/len(vina_score_list)}")
        print(f"Vina score median: {sorted(vina_score_list)[len(vina_score_list)//2]}")
        print(f"Vina min mean: {sum(vina_min_list)/len(vina_min_list)}")
        print(f"Vina min median: {sorted(vina_min_list)[len(vina_min_list)//2]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('-n', '--num_process', type=int, default=1)
    args = parser.parse_args()

    runner = EvaluaionRunner(args)
    runner.run()
