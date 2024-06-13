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


def read_pdbbind(path_root, error_log):
    path_root = Path(path_root)
    sdf_files = path_root.glob("**/*.sdf")
    for sdf_file in tqdm(sdf_files, ncols=80):
        relative_path = sdf_file.relative_to(path_root)
        mol = None
        try:
            mol = parse_mol(sdf_file)
        except Exception:
            pass
        if mol is None:
            try:
                mol2_path = sdf_file.with_suffix(".mol2")
                mol = parse_mol(mol2_path)
            except Exception:
                pass
        if mol is None:
            message = record_error(
                error_log,
                "Failed to parse sdf and mol2",
                src=str(sdf_file),
            )
            logger.error(message)
            continue
        if type(mol) == list:
            for m in mol:
                yield {"mol": m, "src": str(relative_path)}
        else:
            yield {"mol": mol, "src": str(relative_path)}


if __name__ == "__main__":
    path_root = "/data/screening/pdbbind_2020/combine_set"
    lmdb_path = "/data/screening/smilesdb/smilesdb_pdbbind_test.lmdb"
    error_log_path = "/data/screening/smilesdb/pdbbind2020_error_test.log"
    write_smiles_to_lmdb_ray(
        lmdb_path=lmdb_path,
        split_name="pdbbind2020",
        read_sample_func=partial(read_pdbbind, path_root, error_log_path),
        error_log=error_log_path,
    )
