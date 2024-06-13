import argparse
from pathlib import Path

import pandas as pd


def csv2smi(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in input_dir.glob("*.csv"):
        smi_path = output_dir / f"{csv_path.stem}.smi"
        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]
        lines = [f"{line.strip()} {i}\n" for i, line in enumerate(lines)]
        with open(smi_path, "w") as f:
            f.writelines(lines)


def extract_csv2smi(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in input_dir.glob("*.csv"):
        smi_path = output_dir / f"{csv_path.stem}.smi"
        data = pd.read_csv(csv_path)
        smiles = data["smiles"].tolist()
        lines = [f"{smi} {i}\n" for i, smi in enumerate(smiles)]
        with open(smi_path, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # csv2smi(args.input_dir, args.output_dir)
    extract_csv2smi(args.input_dir, args.output_dir)
