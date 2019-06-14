import torch
import torch.nn as nn
from torchvision.models import resnet

from attrdict import AttrDict

from my_imp.models.modules.scene_graph import SceneGraph
import my_imp.models.modules.reasoning_v1.quasi_symbolic as qs

SNG_ARGS = AttrDict({
    'feature_dim': 256,
    'output_dims': [None, 256, 256],
    'downsample_rate': 16,
})

QSDF_ARGS = AttrDict({
    'vse_known_belong': False,
    'vse_large_scale': False,
    'vse_ls_load_concept_embeddings': False,
    'vse_hidden_dims': [None, 64, 64],
})

attribute_concepts = {
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
    'material': ['rubber', 'metal'],
    'shape': ['cube', 'sphere', 'cylinder'],
    'size': ['small', 'large']
}

relational_concepts = {
    'spatial_relation': ['left', 'right', 'front', 'behind']
}

operation_signatures = [
        # Part 1: CLEVR dataset.
        ('scene', [], [], 'object_set'),
        ('filter', ['concept'], ['object_set'], 'object_set'),
        ('relate', ['relational_concept'], ['object'], 'object_set'),
        ('relate_attribute_equal', ['attribute'], ['object'], 'object_set'),
        ('intersect', [], ['object_set', 'object_set'], 'object_set'),
        ('union', [], ['object_set', 'object_set'], 'object_set'),

        ('query', ['attribute'], ['object'], 'word'),
        ('query_attribute_equal', ['attribute'], ['object', 'object'], 'bool'),
        ('exist', [], ['object_set'], 'bool'),
        ('count', [], ['object_set'], 'integer'),
        ('count_less', [], ['object_set', 'object_set'], 'bool'),
        ('count_equal', [], ['object_set', 'object_set'], 'bool'),
        ('count_greater', [], ['object_set', 'object_set'], 'bool'),
    ]

operation_signatures_dict = {v[0]: v[1:] for v in operation_signatures}


class NSCLModel(nn.Module):
    def __init__(
            self,
            num_claasses=28,
            pretrained_resnet=True,
            sng_args=SNG_ARGS,
            qsdf_args=QSDF_ARGS,
        ):
        super().__init__()

        self.sng_args = sng_args
        self.qsdf_args = qsdf_args
        self.num_classes = num_claasses
    
        self.resnet = resnet.resnet34(pretrained=pretrained_resnet)
        self.resnet.layer4 = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.scene_graph = SceneGraph(**sng_args)

        self.reasoning = qs.DifferentiableReasoning(
            self._make_vse_concepts(
                qsdf_args.vse_large_scale, qsdf_args.vse_known_belong),
            self.scene_graph.output_dims, qsdf_args.vse_hidden_dims,
        )

    def _make_vse_concepts(self, large_scale, known_belong):
        # if large_scale:
        #    return {
        #        'attribute_ls': {'attributes': list(gdef.ls_attributes), 'concepts': list(gdef.ls_concepts)},
        #        'relation_ls': {'attributes': None, 'concepts': list(gdef.ls_relational_concepts)},
        #        'embeddings': gdef.get_ls_concept_embeddings()
        #    }
        return {
            'attribute': {
                'attributes': list(attribute_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in attribute_concepts.items() for v in vs
                ]
            },
            'relation': {
                'attributes': list(relational_concepts.keys()) + ['others'],
                'concepts': [
                    (v, k if known_belong else None)
                    for k, vs in relational_concepts.items() for v in vs
                ]
            }
        }

    def forward(self, input):

        monitors, outputs = {}, {}

        f_scene = self.resnet(input['image'])
        # TEMP hardcode dimensions
        # print(f_scene.size())

        f_scene = f_scene.view(input['image'].size(0), self.sng_args.feature_dim, 16, 24)
        
        f_sng = self.scene_graph(f_scene, input['objects'], input['objects_length'])

        programs = input['program_qsseq']
        programs, buffers, answers = self.reasoning(
            f_sng, programs, fd=None)
        outputs['buffers'] = buffers
        outputs['answer'] = answers

        return programs, buffers, answers
