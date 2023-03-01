# Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

import torch

import clip
from clip.model import ModifiedResNet


def load_clip():
    clip_model, clip_transforms = clip.load("RN50")
    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    for b in [1, 2, 3, 4]])
    backbone = ModifiedResNetFeatures(layers)
    backbone.load_state_dict(clip.visual.state_dict())
    transforms = clip_transforms.transforms[-1]
    return backbone, transforms


class ModifiedResNetFeatures(ModifiedResNet):
    def __init__(self, layers, output_dim=1, heads=1, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x0 = stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }
