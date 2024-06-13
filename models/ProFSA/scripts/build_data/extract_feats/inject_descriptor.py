import os
import pickle as pk

import numpy as np
import pandas


def encode_mols(smile_file, out_csv, n_jobs=1):
    os.system(
        "python -m mordred {} -p {} -o {}".format(smile_file, n_jobs, out_csv)
    )
    print("chem descriptors saved to {}".format(out_csv))
    return None


def csv2dict(csv_file):
    # this csv file has a head of 'name,feature1,feature2,...',missing value is '', delete all rows that all features are missing values, and then delete all columns that have missing values. save the result as a dict with key=name, value=feature numpy array
    df = pandas.read_csv(csv_file)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="any")
    df = df.set_index("name")
    df = df.astype(np.float32)
    out = df.to_dict("tight")
    # convert 'data' to numpy array
    out.__delitem__("index_names")
    out.__delitem__("column_names")
    out["data"] = np.array(out["data"])
    return out


if __name__ == "__main__":
    # load data
    test_data = csv2dict(
        "/drug/Docking_Based_MolEncoder_Benchmark/chem_descriptors.csv"
    )
    print(test_data["data"].shape)
    # save data
    with open(
        "/drug/Docking_Based_MolEncoder_Benchmark/chem_descriptors.pkl", "wb"
    ) as f:
        pk.dump(test_data, f)
