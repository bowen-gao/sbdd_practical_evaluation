import sys

from tqdm import trange

sys.path.append(".")  # noqa: E402
from src.dataset.components.lmdb import (  # noqa: E402
    UniMolLMDBDataset as LMDBDataset,
)


def write_key_from_to(src_lmdb, src_key, tgt_lmdb, tgt_key):
    src_dataset = LMDBDataset(src_lmdb, readonly=True)
    tgt_dataset = LMDBDataset(tgt_lmdb, readonly=False)
    # breakpoint()

    assert len(src_dataset) == len(
        tgt_dataset
    ), f"Length mismatch: {len(src_dataset)} vs {len(tgt_dataset)}"

    # for i in range(10):
    for i in trange(len(src_dataset), ncols=80):
        src_data = src_dataset[i]
        tgt_data = tgt_dataset[i]

        assert (
            src_data["smi"] == tgt_data["smi"]
        ), f"SMILES mismatch: {src_data['smi']} vs {tgt_data['smi']}"

        tgt_data[tgt_key] = src_data[src_key]
        tgt_dataset[i] = tgt_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--src_key", type=str, required=True)
    parser.add_argument("--tgt_path", type=str, required=True)
    parser.add_argument("--tgt_key", type=str, required=True)
    args = parser.parse_args()

    write_key_from_to(args.src_path, args.src_key, args.tgt_path, args.tgt_key)
