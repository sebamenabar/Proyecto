import copy
import json
import random
import os.path as osp
from attrdict import AttrDict

import nltk
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import transforms

import jactorch.transforms.bbox as T

from .vocab import Vocab
from .copied.scene_annotation import annotate_objects
from .copied.program_translator import clevr_to_nsclseq, nsclseq_to_nsclqsseq

def gen_image_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # return T.Compose([
    #    T.NormalizeBbox(),
    #    T.Resize(img_size),
    #    T.DenormalizeBbox(),
    #    T.ToTensor(),
    #    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #])

def gen_bbox_transform(img_size):
    transform = T.Compose([
        T.NormalizeBbox(),
        T.Resize(img_size),
        T.DenormalizeBbox(),
    ])
    
    def fun(img, bbox):
        _, bbox = transform(img, bbox)
        return torch.from_numpy(bbox)
        
    return fun

class MyDataset(Dataset):
    def __init__(self, scenes_json, questions_json,
                 image_root, image_transform, bbox_transform,
                 vocab_json, ans_dict_json,
                 question_transform=None, incl_scene=True,
                 sample_size=None, seed=None):
        super().__init__()

        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.bbox_tranform = bbox_transform
        self.vocab_json = vocab_json
        self.question_transform = question_transform
        self.ans_dict_json = ans_dict_json

        self.incl_scene = incl_scene

        print('Loading answers from: "{}"'.format(self.ans_dict_json))
        self.ans = Vocab.from_json(self.ans_dict_json)

        print('Loading scenes from: "{}".'.format(self.scenes_json))
        with open(self.scenes_json, 'r') as f:
            self.scenes = json.load(f)['scenes']

        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                print('Loading questions from: "{}".'.format(filename))
                with open(filename, 'r') as f:
                    self.questions.extend(
                        json.load(f)['questions'])
        else:
            print('Loading questions from: "{}".'.format(
                questions_json))
            with open(self.questions_json, 'r') as f:
                self.questions = json.load(f)[
                    'questions']

        if sample_size:
            if seed:
                np.random.seed(seed)
            self.questions = np.random.choice(self.questions, size=sample_size, replace=False)
            self.scenes = {q['image_index']: self.scenes[q['image_index']] for q in self.questions}

        # print(len(self.questions))
        # print(len(self.scenes))
        # print(self.scenes)

        print('Loading vocab from: "{}".'.format(self.vocab_json))
        self.vocab = Vocab.from_json(self.vocab_json)

        self.prepare_data()

    def prepare_data(self):

        print('Preparing scenes')
        dummy_fp = osp.join(self.image_root, self.scenes[self.questions[0]['image_index']]['image_filename'])
        dummy_image = Image.open(dummy_fp).convert('RGB')
        for i, scene in enumerate(self.scenes.values()):
            # scene = scenes['scenes'][i]
            print(f'\r{i + 1}/{len(self.scenes)}', end='')
            objects = annotate_objects(scene)['objects']
            # scene['objects_raw'] = scene['objects']
            scene['objects'] = self.bbox_tranform(dummy_image, objects)
            scene['scene_size'] = len(scene['objects'])
            scene['image_filename'] = osp.join(self.image_root, scene['image_filename'])

            del scene['relationships']
            del scene['objects_detection']
            del scene['directions']

            # break

        print('\nPreparing questions')
        for i, q in enumerate(self.questions):
            print(f'\r{i + 1}/{len(self.questions)}', end='')
            q['program_size'] = len(q['program'])
            q['question_raw'] = q['question']
            q['question'] = np.array(self.vocab.map_sequence(nltk.word_tokenize(q['question'].lower())), dtype='int32')
            q['answer_raw'] = q['answer']
            q['answer'] = self.ans.word2idx[q['answer']]
            # q['program_raw'] = q['program']
            # q['program_seq'] = clevr_to_nsclseq(q['program'])
            program_seq = clevr_to_nsclseq(q['program'])
            q['program_qsseq'] = nsclseq_to_nsclqsseq(program_seq)
            q['question_type'] = program_seq[-1]['op']

            del q['program']
            del q['image_filename']

            # break
        print()

    def __getitem__(self, index):
        # fd = self.questions[index]
        # scene = self.scenes[fd['image_index']]

        # image = Image.open(scene['image_filename']).convert('RGB')
        # image = self.image_transform(image)

        # Testing without vars because of memory leaks
        return {
            'image': self.image_transform(Image.open(self.scenes[self.questions[index]['image_index']]['image_filename']).convert('RGB')),
            **self.questions[index],
            'objects': self.scenes[self.questions[index]['image_index']]['objects'],
            }

    def __len__(self):
        return len(self.questions)

def collate_fn(batch):
    # batch_size = len(batch)

    images = torch.stack([d['image'] for d in batch])
    answers = torch.tensor([d['answer'] for d in batch], dtype=torch.uint8)
    objects_len = torch.tensor([len(d['objects']) for d in batch], dtype=torch.uint8)
    objects = pad_sequence([d['objects'] for d in batch], batch_first=True)
    questions = pad_sequence([torch.tensor(d['question'], dtype=torch.long) for d in batch], batch_first=True)
    # programs = AttrDict({
    #    'program_raw': [d['program_raw'] for d in batch],
    #    'program_seq': [d['program_seq'] for d in batch],
    #    'program_qsseq': [d['program_qsseq'] for d in batch],
    #})
    program_qsseq = [d['program_qsseq'] for d in batch]

    return AttrDict({
        'image': images,
        'answer': answers,
        'objects_length': objects_len,
        'objects': objects,
        'questions': questions,
        'program_qsseq': program_qsseq,
        'question_type': [d['question_type'] for d in batch],
        'question_raw': [d['question_raw'] for d in batch],
        'answer_raw': [d['answer_raw'] for d in batch],
        # **programs,
    })