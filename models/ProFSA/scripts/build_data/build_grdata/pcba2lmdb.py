import logging
import sys
from functools import partial

import pandas as pd

sys.path.append(".")  # noqa: E402
from src.dataset.components.smiles_writer import (  # noqa: E402
    parse_mol,
    record_error,
    write_smiles_to_lmdb_ray,
)

logger = logging.getLogger(__name__)


def read_pcba(path, error_log):
    data = pd.read_csv(path)
    smiles = data["smiles"].tolist()
    for smi in smiles:
        mol = None
        try:
            mol = parse_mol(smi)
        except Exception:
            pass
        if mol is None:
            message = record_error(
                error_log,
                "Failed to parse smi",
                src=str(smi),
            )
            logger.error(message)
            continue
        yield {"mol": mol}


if __name__ == "__main__":
    path = "/data/screening/pcba.csv"
    lmdb_path = "/data/screening/smilesdb/smilesdb_pcba.lmdb"
    error_log_path = "/data/screening/smilesdb/pcba_error.log"
    write_smiles_to_lmdb_ray(
        lmdb_path=lmdb_path,
        split_name="pcba",
        read_sample_func=partial(read_pcba, path, error_log_path),
        error_log=error_log_path,
    )
