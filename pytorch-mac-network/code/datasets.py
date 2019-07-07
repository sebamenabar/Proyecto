from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import json
import os
import os.path
import pickle
import random
import re
from pathlib import Path

import h5py
import numpy as np
import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pycocotools.mask as mask_utils
from PIL import Image

from config import cfg

initial_size = (320, 480)
final_size = (224, 224)

_transform = transforms.Compose([
    transforms.Resize(final_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def norm_bbox(bbox, initial_size=initial_size, final_size=(56, 56)):
    bbox = bbox.copy()
    bbox[:, 0] = np.floor((bbox[:, 0] / initial_size[1]) * final_size[1])
    bbox[:, 1] = np.floor((bbox[:, 1] / initial_size[0]) * final_size[0])
    bbox[:, 2] = np.ceil((bbox[:, 2] / initial_size[1]) * final_size[1])
    bbox[:, 3] = np.ceil((bbox[:, 3] / initial_size[0]) * final_size[0])

    return bbox.astype(np.int)

class ClevrDataset(data.Dataset):
    def __init__(self, data_dir, img_dir, scenes_json, split='train',
    prepare_scenes=True, use_sample=False, incl_objs=False, raw_image=False):

        self.incl_objs = incl_objs
        if use_sample:
            split = 'train'
            scenes_json = scenes_json.replace('val', 'train')
            

        print('Loading data')
        if use_sample:
            data_file = os.path.join(data_dir, 'sample_{}.pkl'.format(split))
        else:
            data_file = os.path.join(data_dir, '{}.pkl'.format(split))
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        if not raw_image:
            print('Loading features')
            fp_data = os.path.join(data_dir, '{}_features.h5'.format(split))
            if os.path.exists(fp_data):
                self.img = h5py.File(fp_data, 'r')['features']
            else:
                self.img = h5py.File(os.path.join(
                    data_dir, '{}.h5'.format(split)), 'r')['features']
        
        print('Loading scenes')
        with open(scenes_json, 'r') as f:
            self.scenes = json.load(f)['scenes']
        self.imgfile2idx = {
            scene['image_filename']: i for i, scene in enumerate(self.scenes)}
        
        self.img_dir = img_dir
        self.split = split
        self.raw_image = raw_image

        if prepare_scenes:
            self.prepare_scenes()
            

    def prepare_scenes(self):
        print('Preparing scenes')
        for i, scene in enumerate(self.scenes):
            print(f'\r{i + 1}/{len(self.scenes)}', end='')
            # print(scene['objects_detection'])
            if self.split == 'mini':
                detection = 'objects'
            else:
                detection = 'objects_detection'
            boxes = np.array([mask_utils.toBbox(obj['mask'])
                               for obj in scene[detection]])
            scene['boxes'] = norm_bbox(boxes)
        print()

    def __getitem__(self, index):
        family = None
        if self.split == 'mini':
            data = self.data[index]
            if len(data) == 3:
                imgfile, question, answer = data
                attention = None
            else:
                imgfile, question, answer, attention = data
        else:
            attention = None
            imgfile, question, answer, family = self.data[index]
        scene = self.scenes[self.imgfile2idx[imgfile]]

        assert scene['image_filename'] == imgfile

        if self.raw_image:
            img = None
        else:
            id = int(imgfile.rsplit('_', 1)[1][:-4])
            img = torch.from_numpy(self.img[id])

        if self.split == 'mini':
            img_path = os.path.join(self.img_dir, imgfile)
        else:
            img_path = os.path.join(self.img_dir, self.split, imgfile)

        raw_image = Image.open(img_path).convert('RGB')
        raw_image = _transform(raw_image)

        return img, question, len(question), answer, family, torch.from_numpy(scene['boxes']), raw_image, attention

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    # lengths, answers, boxes, raw_images, = [], [], [], []
    images, lengths, answers, boxes, raw_images, attentions = [], [], [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)

    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        # question, length, answer, box, raw_image, attention = b
        image, question, length, answer, family, box, raw_image, attention = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        boxes.append(box)
        raw_images.append(raw_image)
        attentions.append(attention)

    return {'image': image, 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths,
            'raw_image': torch.stack(raw_images), 'boxes': boxes,
            'attention': attentions,
            }
