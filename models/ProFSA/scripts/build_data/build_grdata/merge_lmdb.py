import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(".")  # noqa: E402
from src.dataset.components.lmdb import LMDBDataset  # noqa: E402

logger = logging.getLogger(__name__)


def merge_lmdb(src_path, src_split, tgt_path, tgt_split, error_log_path=None):
    if error_log_path is None:
        error_log_path = Path(tgt_path).parent / "merge_error.log"

    src = LMDBDataset(src_path, src_split)
    tgt = LMDBDataset(tgt_path, tgt_split, readonly=False)
    src_split_keys = src.get_split(src_split)
    tgt_split_keys = tgt.get_split(tgt_split)

    try:
        for key in tqdm(src_split_keys, ncols=80):
            tgt_split_keys.append(key)
            if key in tgt:
                if src[key]["smi"] != tgt[key]["smi"]:
                    error_message = f"Same key but different smi: {key}"
                    logger.warning(error_message)
                    with open(error_log_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "src": str(src_path),
                                    "tgt": str(tgt_path),
                                    "key": key,
                                }
                            )
                        )
            else:
                tgt[key] = src[key]
    except Exception:
        logger.exception("Error occurred")
    finally:
        tgt.set_split(tgt_split, tgt_split_keys)
        logger.info(f"Summary: {tgt.summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_path", type=str)
    parser.add_argument("src_split", type=str)
    parser.add_argument("tgt_path", type=str)
    parser.add_argument("tgt_split", type=str)
    parser.add_argument("--error_log_path", type=str, default=None)
    args = parser.parse_args()
    merge_lmdb(
        args.src_path,
        args.src_split,
        args.tgt_path,
        args.tgt_split,
        args.error_log_path,
    )
