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

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from config import cfg, cfg_from_file
from utils import mkdir_p
from trainer import Trainer

from torch.nn import optim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='shapes_train.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--use_sample', type=bool, default=False)
    args = parser.parse_args()
    return args


def set_logdir(max_steps):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    logdir = "data/{}_max_steps_{}".format(now, max_steps)
    mkdir_p(logdir)
    print("Saving output to: {}".format(logdir))
    code_dir = os.path.join(os.getcwd(), "code")
    mkdir_p(os.path.join(logdir, "Code"))
    for filename in os.listdir(code_dir):
        if filename.endswith(".py"):
            shutil.copy(code_dir + "/" + filename, os.path.join(logdir, "Code"))
    shutil.copy(args.cfg_file, logdir)
    return logdir


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.use_sample:
        cfg.DATASET.USE_SAMPLE = args.use_sample
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if cfg.TRAIN.FLAG:
        logdir = set_logdir(cfg.TRAIN.MAX_STEPS)
        trainer = Trainer(logdir, cfg)

        if cfg.TRAIN.MINI_EPOCHS > 0:
            trainer.train_mini()

        trainer.lr = cfg.TRAIN.LEARNING_RATE
        trainer.optimizer = optim.Adam(trainer.model.parameters(), lr=trainer.lr)

        trainer.previous_best_acc = 0.0
        trainer.previous_best_epoch = 0

        trainer.total_epoch_loss = 0
        trainer.prior_epoch_loss = 10

        trainer.train()
    else:
        raise NotImplementedError

