import sys

from hydra import compose, initialize
from omegaconf import open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.dataset.components.lmdb import LMDBDataset  # noqa: E402
from src.dataset.profsa import NextMolDataset  # noqa: E402
from src.model.drugclip import DrugCLIP  # noqa: E402


def load_model():
    with initialize(version_base="1.3", config_path="../../../conf"):
        cfg = compose(config_name="train")
        with open_dict(cfg):
            cfg.hydra = None
    model = DrugCLIP(cfg.model.cfg).cuda()
    model.train()
    return model


def extract_unimol(lmdb_path, split="full", batch_size=256):
    model = load_model()
    moldb = LMDBDataset(lmdb_path, readonly=False)
    dataset = NextMolDataset(lmdb_path, split=split, return_key=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=64,
        pin_memory=False,
        collate_fn=dataset.collater,
    )
    for batch in tqdm(dataloader, ncols=80):
        keys = batch["key"]
        rep = model.forward_mol(
            batch["mol_src_tokens"].cuda(),
            batch["mol_src_distance"].cuda(),
            batch["mol_src_edge_type"].cuda(),
            return_rep=True,
        )
        for key, vector in zip(keys, rep):
            sample = moldb[key]
            sample["unimol"] = vector.cpu().numpy()
            moldb[key] = sample


if __name__ == "__main__":
    lmdb_path = "/data/screening/smilesdb/smilesdb.lmdb"
    model = load_model()
    extract_unimol(lmdb_path=lmdb_path)
