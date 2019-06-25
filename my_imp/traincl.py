import yaml
from attrdict import AttrDict
from my_imp.models.nscl import NSCLModel
from my_imp.data import collate_fn
from my_imp.data import DatasetV2, gen_image_transform, gen_bbox_transform

import sys
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        cfg = yaml.safe_load(f)

    bbox_transform = gen_bbox_transform(cfg.img_size)
    image_transform = gen_image_transform(cfg.img_size)
    
    train_ds = DatasetV2(
        ans_dict_json=cfg.ans_dict_json,
        image_transform=image_transform,
        bbox_transform=bbox_transform,
        **cfg.train_ds,
    )
    val_ds = DatasetV2(
        ans_dict_json=cfg.ans_dict_json,
        image_transform=image_transform,
        bbox_transform=bbox_transform,
        **cfg.val_ds,
    )

