import os
from tqdm import tqdm

from src.dataset.fungi import FungiTastic
from src.dataset.utils.fungi_vis import FungiTasticVis

from src.utils import load_yaml


def _extract_paths_and_labels(ds):
    """Extract file paths and labels."""
    paths, labels = [], []
    for i in tqdm(range(len(ds)), desc="Indexing"):
        _, y, p = ds[i]  # (PIL_image, label, path)

        paths.append(str(p))
        if y:
            labels.append(int(y))
    return paths, labels


def read_img_paths_to_train_val_set(cfg_path, random_state=0, transform=None):
    '''Reads image paths from config into FungiTastic sets.'''
    config_ = load_yaml(cfg_path)
    
    # take 1st level common params
    img_cfg = config_['image']
    take_fraction = config_['take_fraction']
    train_test_split = config_['train_val_split']
    
    # take img processing specific params
    root_img_path = img_cfg['datas_mini_path']
    size = img_cfg['size']
    shuffle = img_cfg['shuffle']
    
    root_img_path = f"{os.getcwd()}{root_img_path}"
    
    trainset, valset = FungiTastic.from_path_train_test(
        root=root_img_path,
        data_subset='Mini',
        split='train',             # which CSV to read from (Train/Val/Test filenames)
        size=size,
        task='closed',
        take_fraction=take_fraction,       # take first {fraction}% of rows from the CSV
        train_test_split=train_test_split,
        shuffle=shuffle,            # optional: shuffle before splitting
        random_state=random_state,          # reproducible shuffle
        transform=transform
    )
    
    trainset_paths, trainset_labels = _extract_paths_and_labels(trainset)
    valset_paths, valset_labels = _extract_paths_and_labels(valset)

    return trainset_paths, trainset_labels, valset_paths, valset_labels


def read_img_paths_to_test_set(cfg_path, transform=None):
    '''Reads image paths from config into FungiTastic sets.'''
    config_ = load_yaml(cfg_path)
    img_cfg = config_['image']
    root_img_path = img_cfg['datas_mini_path']
    size = img_cfg['size']

    testset = FungiTasticVis(
        root=root_img_path,
        split='val',
        size=size,
        task='closed',
        data_subset='Mini',
        transform=transform,
    )
    
    return testset