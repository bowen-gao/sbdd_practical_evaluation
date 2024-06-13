import os
import sys
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from multiprocessing import Pool
from Bio.PDB import PDBParser, Selection, PDBIO
from Bio.PDB.Polypeptide import is_aa
from dataset import BioLipDataset, CrossDockedDataset, DUDEDataset, Dekois2Dataset, PCBADataset, PDBBindDataset

class PocketExtractor():
    
    def __init__(self,dataset):
        self.dataset = dataset

    def run(self, distance=20.0):
        tasks = list(self.dataset.get_items())
        tasks = [(x, distance) for x in tasks]


        # Use pool.map to process tasks in parallel
        with Pool(50) as p:
            results = p.starmap(self._extract_single, tasks)
    
    def _extract_single(self,item,distance):
        # print(item)
        protein_dir = item['protein_dir']
        ligand_dir = item['ligand_dir']
        # print(f"Extract pocket from {protein_dir} with ligand {ligand_dir}")

        pocket_dir = item['pocket20A_dir']
        # if os.path.exists(pocket_dir):
        #     return

        # read ligand
        try:
            if ligand_dir.endswith('.mol2'):
                ligand = Chem.MolFromMol2File(ligand_dir,sanitize=False)
            elif ligand_dir.endswith('.pdb'):
                ligand = Chem.MolFromPDBFile(ligand_dir,sanitize=False)
            elif ligand_dir.endswith('.sdf'):
                ligand = Chem.MolFromMolFile(ligand_dir,sanitize=False)
            else:
                raise NotImplementedError
            assert ligand is not None
        except:
            print(f"Failed to read ligand {ligand_dir}")
            return
            
        conf = ligand.GetConformer()
        ligand_coords = conf.GetPositions()
        
        # read protein
        try:
            protein = PDBParser(QUIET=True).get_structure("protein",protein_dir)[0]
            assert protein is not None
        except:
            print(f"Failed to read protein {protein_dir}")
            return
        
        # extract pocket
        for chain in protein:
            remove_residue_ids=[]
            for residue in chain:
                f=1
                for atom in residue:
                    protein_atom_coords = np.array(atom.coord)
                    for ligand_coord in ligand_coords:
                        if np.linalg.norm(protein_atom_coords - ligand_coord) < distance:
                            f=0
                            break
                if f:
                    remove_residue_ids.append(residue.id)
            for residue_id in remove_residue_ids:
                chain.detach_child(residue_id)
            
        self.pocket = protein
        # fix non-standard residue
        self._fix_some_non_standard_residue()
        self._remove_all_non_standard_residue()

        # remove all atoms with element X
        self._remove_atom_element_X()

        # save pocket
        io = PDBIO()
        io.set_structure(protein)
        io.save(pocket_dir)
    
    def  _remove_atom_element_X(self):
        for chain in self.pocket:
            for residue in chain:
                remove_atom_ids = []
                for atom in residue:
                    if atom.element == 'X':
                        print(f"Remove atom element X: {atom}")
                        remove_atom_ids.append(atom.id)
                for atom_id in remove_atom_ids:
                    residue.detach_child(atom_id)
    
    def _remove_all_non_standard_residue(self):
        for chain in self.pocket:
            remove_residue_ids = []
            for residue in chain:
                residue_name = residue.get_resname()
                if not is_aa(residue_name, standard=True):
                    print(f"Remove non-standard residue: {residue}")
                    remove_residue_ids.append(residue.id)
            for residue_id in remove_residue_ids:
                chain.detach_child(residue_id)

    def _fix_some_non_standard_residue(self):
        for chain in self.pocket:
            for residue in chain:
                residue_name = residue.get_resname()
                if residue_name == 'HIE':
                    residue.resname = 'HIS'
                if residue_name == 'HID':
                    residue.resname = 'HIS'
                if residue_name == 'GLZ':
                    residue.resname = 'GLY'
                if residue_name == 'LEV':
                    residue.resname = 'LEU'
                if residue_name == 'SEM':
                    residue.resname = 'MET'
                if residue_name == 'MEU':
                    residue.resname = 'MET'
                if residue_name == 'HIZ':
                    residue.resname = 'HIS'
                if residue_name == 'HIP':
                    residue.resname = 'HIS'
                if residue_name == 'CYX':
                    residue.resname = 'CYS'
                if residue_name == 'DIC':
                    residue.resname = 'ASP'
                if residue_name == 'ASH':
                    residue.resname = 'ASP'
                if residue_name == 'GLV':
                    residue.resname = 'GLY'
                if residue_name == 'HIP':
                    residue.resname = 'HIS'
                if residue_name == 'CYM':
                    residue.resname = 'CYS'
                if residue_name == 'GLO':
                    residue.resname = 'GLU'
                if residue_name == 'ALB':
                    residue.resname = 'ALA'
                if residue_name == 'HIY':
                    residue.resname = 'HIS'
                if residue_name == 'ASZ':
                    residue.resname = 'ASP'
                if residue_name == 'CYT':
                    residue.resname = 'CYS'
                if residue_name == 'DID':
                    residue.resname = 'ASP'
                if residue_name == 'TYM':
                    residue.resname = 'TYR'
                if residue_name == 'ASM':
                    residue.resname = 'ASP'
                if residue_name == 'SAM':
                    residue.resname = 'MET'
                if residue_name == 'GLM':
                    residue.resname = 'GLU'
                if residue_name == 'ASQ':
                    residue.resname = 'ASP'
                
                    

if __name__ == "__main__":
    dataset = PCBADataset()
    pocket_extractor = PocketExtractor(dataset)
    pocket_extractor.run()

