import sys

from rdkit.Chem import AllChem as Chem
from tqdm import trange

sys.path.append(".")  # noqa: E402
from src.dataset.components.lmdb import (  # noqa: E402
    UniMolLMDBDataset as LMDBDataset,
)


def gen_mol_files(db_path: str, smi_path: str, sdf_path: str = None):
    dataset = LMDBDataset(db_path)
    smi_writer = Chem.SmilesWriter(smi_path, includeHeader=False)

    if sdf_path:
        sdf_writer = Chem.SDWriter(sdf_path)

    # for i in trange(100, ncols=80):
    for i in trange(len(dataset), ncols=80):
        mol = Chem.MolFromSmiles(dataset[i]["smi"])
        mol.SetProp("_Name", str(i))

        smi_writer.write(mol)

        if sdf_path:
            mol = Chem.AddHs(mol)
            Chem.EmbedMolecule(mol, randomSeed=2024)
            Chem.MMFFOptimizeMolecule(mol)
            sdf_writer.write(mol)

        if i % 1000 == 0:
            smi_writer.flush()
            if sdf_path:
                sdf_writer.flush()

    smi_writer.close()
    if sdf_path:
        sdf_writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--smi_path", type=str, required=True)
    parser.add_argument("--sdf_path", type=str, default=None)
    args = parser.parse_args()

    gen_mol_files(args.db_path, args.smi_path, args.sdf_path)
