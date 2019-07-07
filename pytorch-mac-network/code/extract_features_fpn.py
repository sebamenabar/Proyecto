# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json
import h5py
import numpy as np
# from scipy.misc import imread, imresize
from PIL import Image

import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision import transforms

from collections import OrderedDict

import torch.utils.data as data

from PIL import Image
import os
import os.path

from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm


def default_loader(path):
	return Image.open(path).convert('RGB')


class ImageFilelist(data.Dataset):
    def __init__(self, flist, transform=None, loader=default_loader):
        self.flist = flist
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
      impath = self.flist[index]
      img = self.loader(impath[0])
      if self.transform is not None:
        img = self.transform(img)

      # print(index)
      return img

    def __len__(self):
        return len(self.flist)


parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', required=True)

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=128, type=int)


def build_model(args):
  if not hasattr(torchvision.models, args.model):
    raise ValueError('Invalid model "%s"' % args.model)
  if not 'resnet' in args.model:
    raise ValueError('Feature extraction only supports ResNets')
  cnn = getattr(torchvision.models, args.model)(pretrained=True, norm_layer=FrozenBatchNorm2d)
  layers = OrderedDict([
    ('conv1', cnn.conv1),
    ('bn1', cnn.bn1),
    ('relu', cnn.relu),
    ('maxpool', cnn.maxpool),
  ])
  for i in range(4):
      name = 'layer%d' % (i + 1)
      layers[name] = getattr(cnn, name)
  model = torch.nn.Sequential(layers)
  model.cuda()
  model.eval()

  # print(model)

  return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

  model = IntermediateLayerGetter(model, return_layers=return_layers)

  model.cuda()
  model.eval()
  return model


def run_batch(cur_batch, model):
  # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  # std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

  # image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
  # image_batch = (image_batch / 255.0 - mean) / std
  # image_batch = torch.FloatTensor(image_batch).cuda()
  # image_batch = torch.autograd.Variable(image_batch, volatile=True)

  with torch.no_grad():
      feats = model(cur_batch)
    # feats = feats.data.cpu().clone().numpy()

  return feats


def main(args):
  input_paths = []
  idx_set = set()
  for fn in os.listdir(args.input_image_dir):
    if not fn.endswith('.png'):
      continue
    idx = int(os.path.splitext(fn)[0].split('_')[-1])
    input_paths.append((os.path.join(args.input_image_dir, fn), idx))
    idx_set.add(idx)
  input_paths.sort(key=lambda x: x[1])
  assert len(idx_set) == len(input_paths)
  assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
  if args.max_images is not None:
    input_paths = input_paths[:args.max_images]
  print(input_paths[0])
  print(input_paths[-1])

  model = build_model(args)

  img_size = (args.image_height, args.image_width)
  transform = transforms.Compose([
    transforms.Resize(img_size, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]),
  ])

  dataset = ImageFilelist(input_paths, transform=transform)
  loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

  with h5py.File(args.output_h5_file, 'w') as f:

    N = len(input_paths)
    # _, C, H, W = feats.shape
    feat_dset56 = f.create_dataset('features56', (N, 256, 56, 56),
                    dtype=np.float32, compression="gzip", compression_opts=1)
    # feat_dset28 = f.create_dataset('features28', (N, 512, 28, 28),
    #                dtype=np.float32, compression="gzip", compression_opts=1)
    feat_dset14 = f.create_dataset('features14', (N, 1024, 14, 14),
                    dtype=np.float32, compression="gzip", compression_opts=1)
    feat_dset7 = f.create_dataset('features7', (N, 2048, 7, 7),
                    dtype=np.float32, compression="gzip", compression_opts=1)

    i0 = 0
    pbar = tqdm(loader)
    for cur_batch in pbar:
      cur_batch = cur_batch.cuda()
      feats = run_batch(cur_batch, model)
      i1 = i0 + len(cur_batch)
      # print(i0, i1)
      feat_dset56[i0:i1] = feats[0].cpu().clone().numpy()
      # feat_dset28[i0:i1] = feats[1].cpu().clone().numpy()
      feat_dset14[i0:i1] = feats[2].cpu().clone().numpy()
      feat_dset7[i0:i1] = feats[3].cpu().clone().numpy()
      i0 = i1

    # cur_batch = []
    # for i, (path, idx) in enumerate(input_paths):
    #   # img = imread(path, mode='RGB')
    #   img = Image.open(path).convert('RGB')
    #   # img = imresize(img, img_size, interp='bicubic')
    #   img = img.resize(img_size, resample=Image.BICUBIC)
    #   img = np.array(img)
    #   img = img.transpose(2, 0, 1)[None]
    #   cur_batch.append(img)
    #   if len(cur_batch) == args.batch_size:
    #     feats = run_batch(cur_batch, model)
    #     if feat_dset56 is None:
    #       N = len(input_paths)
    #       # _, C, H, W = feats.shape
    #       feat_dset56 = f.create_dataset('features56', (N, 256, 56, 56),
    #                                    dtype=np.float32, compression="gzip", compression_opts=9)
    #       feat_dset28 = f.create_dataset('features28', (N, 512, 28, 28),
    #                                    dtype=np.float32, compression="gzip", compression_opts=9)
    #       feat_dset14 = f.create_dataset('features14', (N, 1024, 14, 14),
    #                                    dtype=np.float32, compression="gzip", compression_opts=9)
    #       feat_dset7 = f.create_dataset('features7', (N, 2048, 7, 7),
    #                                    dtype=np.float32, compression="gzip", compression_opts=9)

    #     i1 = i0 + len(cur_batch)
    #     feat_dset56[i0:i1] = feats[0].cpu().clone().numpy()
    #     feat_dset28[i0:i1] = feats[1].cpu().clone().numpy()
    #     feat_dset14[i0:i1] = feats[2].cpu().clone().numpy()
    #     feat_dset7[i0:i1] = feats[3].cpu().clone().numpy()
    #     i0 = i1
    #     print('Processed %d / %d images' % (i1, len(input_paths)))
    #     cur_batch = []
    # if len(cur_batch) > 0:
    #   feats = run_batch(cur_batch, model)
    #   i1 = i0 + len(cur_batch)
    #   feat_dset56[i0:i1] = feats[0].cpu().clone().numpy()
    #   feat_dset28[i0:i1] = feats[1].cpu().clone().numpy()
    #   feat_dset14[i0:i1] = feats[2].cpu().clone().numpy()
    #   feat_dset7[i0:i1] = feats[3].cpu().clone().numpy()
    # print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
