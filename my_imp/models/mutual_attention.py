import torch
import torch.nn as nn
from torchvision.models import resnet

import math
from attrdict import AttrDict

from my_imp.models.nscl import NSCLModel


SNG_ARGS = AttrDict({
    'feature_dim': 256,
    'output_dims': [None, 256, 256],
    'downsample_rate': 16,
})

class StandardBlock(nn.Module):
    def __init__(self,
                 dim=256,
                 dropout=0.5,
                 num_heads=8,
                 ):
        super().__init__()
        self.dim = 256
        self.num_heads = 8

        self.intra_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
        )
        self.inter_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
        )

        self.ff1 = nn.Linear(dim, dim)
        self.ff2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._norm1 = nn.LayerNorm(dim)
        self._norm2 = nn.LayerNorm(dim)
        self._norm3 = nn.LayerNorm(dim)

    def forward(self, intra, inter, inter_mask=None):

        attn_values, _ = self.inter_attention(
            query=intra,
            key=inter.expand(self.num_heads, *inter.size()),
            value=inter.expand(self.num_heads, *inter.size()),
            key_padding_mask=inter_mask,
        )
        intra = self._norm1(intra + self.dropout(attn_values))

        attn_values, _ = self.intra_attention(
            query=intra,
            key=intra.expand(self.num_heads, *intra.size()),
            value=intra.expand(self.num_heads, *intra.size()),
            key_padding_mask=inter_mask,
        )
        intra = self._norm2(intra + self.dropout(attn_values))

        intra = self._norm3(
            intra + self.dropout(self.ff2(self.relu(self.ff1(intra)))))

        return intra


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model=256, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, d_model=256, vocab=50, dropout=0.5):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, )
        self.d_model = d_model
        # self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class ASR(nn.Module):
    def __init__(self,
                 vocab,
                 num_classes=28,
                 N=10,
                 dim=256,
                 num_heads=8,
                 dropout=0.5,
                 sng_args=SNG_ARGS,
                 ):
        super().__init__()

        self.vocab = vocab

        self.embeddings = Embeddings(
            d_model=dim, dropout=dropout, vocab=len(vocab))
        self.pe = PositionalEncoding(d_model=dim, dropout=dropout)

        self.resnet = resnet.resnet34(pretrained=True)
        self.resnet.layer4 = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        self.scene_graph = SceneGraph(**sng_args)

        self.visual_layers = nn.ModuleList(
            [StandardBlock(dim=dim, num_heads=num_heads, dropout=dropout)
             for _ in range(N)]
        )
        self.question_layers = nn.ModuleList(
            [StandardBlock(dim=dim, num_heads=num_heads, dropout=dropout)
             for _ in range(N)]
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.linear = nn.Linear(2 * dim, num_classes)

    def forward(self, image, objects, objects_length, questions):

        f_scene = self.resnet(image)
        # TEMP hardcoded dimensions
        f_scene = f_scene.view(image.size(
            0), self.sng_args.feature_dim, 16, 24)
        objects = self.scene_graph(f_scene, objects, objects_length)
        print(objects.size())

        questions = self.pe(self.embeddings(questions))
        print(questions.size())
        # questions = questions.permute(1, 0, -1)

        return

        prev_objects = objects
        prev_questions = questions
        for v, q in zip(self.visual_layers, self.question_layers):
            objects = v(objects, prev_questions)
            questions = q(questions, prev_objects)

            prev_objects = objects
            prev_questions = questions

        objects = self.maxpool(objects.permute(1, 2, 0)).squeeze(2)
        questions = self.maxpool(questions.permute(1, 2, 0)).squeeze(2)

        ans = self.linear(torch.cat((objects, questions), dim=1))

        return ans
