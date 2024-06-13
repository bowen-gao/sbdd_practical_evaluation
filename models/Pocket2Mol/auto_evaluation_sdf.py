import os
import argparse
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor
import time
import concurrent.futures
import torch
import rdkit
from rdkit import Chem
from tqdm import tqdm
from evaluation.docking_vina import VinaDockingTask


class EvaluaionRunner:
    def __init__(self, args):
        self.exp_prefix="/data/pocket2mol_data/DUD-E_sample_output/"
        self.protein_root_prefix="/data/DUD-E/raw"
        self.exp_name = args.exp_name
        self.num_process = args.num_process

    def evaluate_case(self, cases):
        for case in cases:
            print(f"Start evaluating {case}")
            case_name=os.path.basename(case).split("_")[-1]
            
            # dude
            protein_root = os.path.join(self.protein_root_prefix, case_name , "receptor.pdb")

            # pcba
            # protein_root = os.path.join(self.protein_root_prefix, case_name)
            # print(protein_root)
            # protein_root = glob.glob(protein_root+"/*_pocket10A.pdb")[0]
            
            sdf_path = case+".sdf"
            
            print(protein_root)
            print(sdf_path)

            results=[]

            reader=Chem.SDMolSupplier(sdf_path)
            for i, mol in tqdm(enumerate(reader)):
                if mol is None:
                    print("Error: %s"%sdf_path)
                    continue

                try:
                    vina_task = VinaDockingTask.from_generated_mol(
                            mol, protein_root=protein_root,
                            tmp_dir="/data/vina_tmp"
                            )
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=8)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=8)
                    
                    results.append({
                        'mol': mol,
                        'vina_score': score_only_results,
                        'vina_min': minimize_results,
                    })
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print('Failed %d' % i)
                    continue

            result_path = os.path.join(case, 'results.pt')
            torch.save(results, result_path)


            print(f"Finish evaluating {case}")

    def run(self):
        case_list = glob.glob(self.exp_prefix + self.exp_name + "/*")
        case_list.sort()
        case_list=[x for x in case_list if os.path.isdir(x)]
        tasks=case_list
        print(f"Total number of cases: {len(case_list)}")

        # for task in tasks[1:]:
        #     self.evaluate_case([task])


        task_splits = [tasks[i::self.num_process] for i in range(self.num_process)]
        print("tasks_splits",task_splits)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_process) as executor:
            futures = []
            for i in range(self.num_process):
                future = executor.submit(self.evaluate_case, task_splits[i])
                futures.append(future)
                time.sleep(0.1)

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
    parser.add_argument('--exp_name', type=str, default='PDBBind-DUD_E_FLAPP_0.9_used_sample_all')
    parser.add_argument('-n', '--num_process', type=int, default=101)
    args = parser.parse_args()

    runner = EvaluaionRunner(args)
    runner.run()
