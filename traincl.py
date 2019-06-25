
from my_imp import initjac
import yaml
from attrdict import AttrDict
from my_imp.models.nscl import NSCLModel
from my_imp.data import collate_fn
from my_imp.data import DatasetV2, gen_image_transform, gen_bbox_transform
from my_imp.copied.losses.reasoning_losses import QALoss
from my_imp.utils import build_curriculum

import sys
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from jactorch.utils.meta import as_tensor, as_float, as_cpu


def default_reduce_func(k, v):
    if torch.is_tensor(v):
        return v.mean()
    
def canonize_monitors(monitors):
    for k, v in monitors.items():
        if isinstance(monitors[k], list):
            if isinstance(monitors[k][0], tuple) and len(monitors[k][0]) == 2:
                monitors[k] = sum([a * b for a, b in monitors[k]]) / max(sum([b for _, b in monitors[k]]), 1e-6)
            else:
                monitors[k] = sum(v) / max(len(v), 1e-3)
        if isinstance(monitors[k], float):
            monitors[k] = torch.tensor(monitors[k])


def update_from_loss_module(monitors, output_dict, loss_update):
    tmp_monitors, tmp_outputs = loss_update
    monitors.update(tmp_monitors)
    output_dict.update(tmp_outputs)

    
reduce_func = default_reduce_func

def step_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch,
    ):

    non_blocking = True
    total_loss = 0.
    total_acc_qa = 0.

    pbar = tqdm(loader)
    for i, feed_dict in enumerate(pbar):
        feed_dict['image'] = feed_dict['image'].to(device, non_blocking=non_blocking)
        feed_dict['objects'] = feed_dict['objects'].to(device, non_blocking=non_blocking)
        feed_dict['objects_length'] = feed_dict['objects_length'].to(device, non_blocking=non_blocking)
        feed_dict['questions'] = feed_dict['questions'].to(device, non_blocking=non_blocking)
        feed_dict['answer'] = feed_dict['answer'].to(device, non_blocking=non_blocking)

        programs, buffers, answers = model(feed_dict)
        loss = criterion(feed_dict, answers)

        monitors = {}
        outputs = {
            'buffers': buffers,
            'answer': answers,
        }
        update_from_loss_module(monitors, outputs, loss)
        canonize_monitors(monitors)

        loss = monitors['loss/qa']

        loss = reduce_func('loss', loss)
        monitors = {k: reduce_func(k, v) for k, v in monitors.items()}

        loss_f = as_float(loss)
        monitors_f = as_float(monitors)

        total_loss += loss_f
        total_acc_qa += monitors_f['acc/qa']

        optimizer.zero_grad()
        optimizer.step()

        pbar.set_postfix(
            loss=f'{loss_f:.4f} ({total_loss/(i + 1):.4f})',
            acc_qa=f'{monitors_f["acc/qa"]} ({total_acc_qa/(i + 1):.4f})',
        )
        pbar.update()

    return total_loss / (i + 1), total_loss / (i + 1)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    device = 'cuda' if cfg.use_cuda and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print('Using GPU backend')

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

    model = NSCLModel().to(device)
    # trainable_parameters = filter(lambda x: x.required_grad, model.parameteres())
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    criterion = QALoss(add_supervision=True)


    for epoch in tqdm(range(1, 120)):

        subset = build_curriculum(train_ds, epoch)
        train_loader = DataLoader(
            subset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            )

        step_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
        )


