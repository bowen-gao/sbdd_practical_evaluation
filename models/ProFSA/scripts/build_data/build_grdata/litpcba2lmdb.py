import itertools
import logging
import sys
from functools import partial
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.dataset.components.smiles_writer import (  # noqa: E402
    parse_mol,
    record_error,
    write_smiles_to_lmdb_ray,
)

logger = logging.getLogger(__name__)


def read_litpcba(path_root, error_log):
    path_root = Path(path_root)
    active_smi_files = path_root.glob("**/actives.smi")
    decoy_smi_file = path_root.glob("**/inactives.smi")
    for smi_file in (
        pbar := tqdm(itertools.chain(active_smi_files, decoy_smi_file))
    ):
        pbar.set_description(f"{smi_file}")
        relative_path = smi_file.relative_to(path_root)
        mol = None
        try:
            mol = parse_mol(smi_file)
        except Exception:
            pass
        if mol is None:
            message = record_error(
                error_log,
                "Failed to parse smi",
                src=str(smi_file),
            )
            logger.error(message)
            continue
        if type(mol) == list:
            for m in mol:
                yield {"mol": m, "src": str(relative_path)}
        else:
            yield {"mol": mol, "src": str(relative_path)}


if __name__ == "__main__":
    path_root = "/data/screening/LIT-PCBA"
    lmdb_path = "/data/screening/smilesdb/smilesdb_litpcba.lmdb"
    error_log_path = "/data/screening/smilesdb/litpcba_error.log"
    write_smiles_to_lmdb_ray(
        lmdb_path=lmdb_path,
        split_name="litpcba",
        read_sample_func=partial(read_litpcba, path_root, error_log_path),
        error_log=error_log_path,
    )
