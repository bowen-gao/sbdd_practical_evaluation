import argparse
import pickle as pkl
from pathlib import Path

import pandas as pd


def desc2pkl(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in input_dir.glob("*_desc.csv"):
        name = csv_path.stem
        pkl_path = output_dir / f"{name}.pkl"
        df = pd.read_csv(csv_path)
        print(f"dimension of {name}: {df.shape}")
        df.dropna(axis=1, how="any", inplace=True)
        print(f"dimension of {name} after dropping NA: {df.shape}")
        df = df.astype("float32")
        df = df.to_numpy()
        with open(pkl_path, "wb") as f:
            pkl.dump(df, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    desc2pkl(args.input_dir, args.output_dir)
    """
    dimension of bace_desc: (1513, 1614)
    dimension of bace_desc after dropping NA: (1513, 1392)
    dimension of esol_desc: (1128, 1614)
    dimension of esol_desc after dropping NA: (1128, 1074)
    dimension of freesolv_desc: (642, 1614)
    dimension of freesolv_desc after dropping NA: (642, 1074)
    dimension of lipophilicity_desc: (4200, 1614)
    dimension of lipophilicity_desc after dropping NA: (4200, 998)
    dimension of sider_desc: (1427, 1614)
    dimension of sider_desc after dropping NA: (1427, 722)
    dimension of tox21_desc: (7831, 1614)
    dimension of tox21_desc after dropping NA: (7831, 687)
    """
