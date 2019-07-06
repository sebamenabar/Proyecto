from __future__ import print_function
import torch

import argparse
import os
import random
import sys
import datetime
import dateutil
import dateutil.tz
import shutil
from tqdm import tqdm

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from config import cfg, cfg_from_file

from torch.utils.data import DataLoader
from datasets import ClevrDataset, collate_fn



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='shapes_train.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--use_sample', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.use_sample:
        cfg.DATASET.USE_SAMPLE = args.use_sample

    img_dir = cfg.DATASET.IMG_DIR
    use_sample = cfg.DATASET.USE_SAMPLE
    data_dir = cfg.DATASET.DATA_DIR
    incl_objs = cfg.TRAIN.RECV_OBJECTS

    if incl_objs:
        print('Including objects')
        collate_fn.incl_objs = True

    print('Using sample', use_sample)

    # load dataset
    train_scenes_json = cfg.DATASET.TRAIN_SCENES_JSON
    dataset = ClevrDataset(data_dir=data_dir, img_dir=img_dir, scenes_json=train_scenes_json, split="train", use_sample=use_sample)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.WORKERS, drop_last=True, collate_fn=collate_fn)

    val_scenes_json = cfg.DATASET.VAL_SCENES_JSON
    dataset_val = ClevrDataset(data_dir=data_dir, img_dir=img_dir,
                                    scenes_json=val_scenes_json, split="val", use_sample=use_sample)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=200, drop_last=True,
                                    shuffle=False, num_workers=cfg.WORKERS, collate_fn=collate_fn)

    print("Size of train dataset: {}".format(len(dataset)))
    print("\n")
    print("Size of val dataset: {}".format(len(dataset_val)))


    for _ in tqdm(dataloader):
        pass

    for _ in tqdm(dataloader_val):
        pass

