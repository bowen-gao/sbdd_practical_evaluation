import logging
from pathlib import Path

import pytest

from src.dataset.components.lmdb import LMDBDataset

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("compress", [False, True])
def test_lmdb_dataset(tmpdir: Path, compress):
    dataset = LMDBDataset(
        lmdb_path=tmpdir / "test.lmdb", compresed=compress, readonly=False
    )

    data = {
        "a": {"name": "a", "value": 1},
        "b": {"name": "b", "value": 2},
        "c": {"name": "c", "value": 3},
    }

    for key, value in data.items():
        dataset[key] = value

    dataset.set_split("train", list(data.keys()))
    dataset.update_full_split()

    for key in data:
        assert dataset[key] == data[key]

    assert dataset.get_split("train") == list(data.keys())
    assert len(dataset) == len(data)
    for key in data:
        assert key in dataset

    logger.info(f"Dataset: {dataset}")
    logger.info(dataset.summary)
