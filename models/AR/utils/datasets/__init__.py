import torch
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset
import os
import pickle

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(
            root,
            data_split_path=config.data_split_path,
            id_split_path=config.id_split_path,
            index_path=config.index_path,

            dataset_name=config.dataset_name,
            split_name=config.split_name,

            *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'id_split_path' in config and config.id_split_path is not None:
        id_split_file=os.path.join(config.id_split_path, config.split_name + '.pkl')
        with open(id_split_file, 'rb') as f:
            split = pickle.load(f)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        print("subsets",subsets.keys())
        print("subsets[test]",subsets["test"])
        return dataset, subsets
    else:
        return dataset
