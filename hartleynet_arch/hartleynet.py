"""HartleyNet architecture.

ResNet-50 backbone with Hartley spectral pooling layers swapped in for
max-pool and the strided convs inside each bottleneck. Single-logit head
trained with sigmoid focal loss; sigmoid(logit) is the probability that
the input image is AI-generated.

The spectral pooling primitives in this file (``_spectral_crop``,
``_spectral_pad``, ``DiscreteHartleyTransform``, ``SpectralPoolingFunction``,
``HartleyPool2d``) are adapted from the reference implementation by
Zhang et al., "Hartley Spectral Pooling for Deep Learning"
(https://github.com/AlbertZhangHIT/HartleySpectralPooling), inlined here
so the demo is self-contained.
"""

import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torchvision.models import resnet50, ResNet50_Weights


NUM_CHANNELS = 3
DROPOUT_RATE = 0.5


def _spectral_crop(x, oh, ow):
    ch = math.ceil(oh / 2)
    cw = math.ceil(ow / 2)

    h_odd = oh % 2 == 1
    w_odd = ow % 2 == 1

    top_h = slice(None, ch)
    bot_h = slice(-(ch - 1), None) if h_odd else slice(-ch, None)
    left_w = slice(None, cw)
    right_w = slice(-(cw - 1), None) if w_odd else slice(-cw, None)

    top = torch.cat((x[:, :, top_h, left_w], x[:, :, top_h, right_w]), dim=-1)
    bot = torch.cat((x[:, :, bot_h, left_w], x[:, :, bot_h, right_w]), dim=-1)
    return torch.cat((top, bot), dim=-2)


def _spectral_pad(reference, output, oh, ow):
    ch = math.ceil(oh / 2)
    cw = math.ceil(ow / 2)

    h_odd = oh % 2 == 1
    w_odd = ow % 2 == 1

    top_h = slice(None, ch)
    bot_h = slice(-(ch - 1), None) if h_odd else slice(-ch, None)
    left_w = slice(None, cw)
    right_w = slice(-(cw - 1), None) if w_odd else slice(-cw, None)

    pad = torch.zeros_like(reference)
    pad[:, :, top_h, left_w] = output[:, :, top_h, left_w]
    pad[:, :, top_h, right_w] = output[:, :, top_h, right_w]
    pad[:, :, bot_h, left_w] = output[:, :, bot_h, left_w]
    pad[:, :, bot_h, right_w] = output[:, :, bot_h, right_w]
    return pad


def DiscreteHartleyTransform(x):
    fft = torch.fft.fft2(x.float(), norm="ortho")
    return fft.real - fft.imag


class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, x, oh, ow):
        ctx.oh = oh
        ctx.ow = ow
        ctx.save_for_backward(x)
        dht = DiscreteHartleyTransform(x)
        cropped = _spectral_crop(dht, oh, ow)
        return DiscreteHartleyTransform(cropped)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        dht = DiscreteHartleyTransform(grad_output)
        padded = _spectral_pad(x, dht, ctx.oh, ctx.ow)
        return DiscreteHartleyTransform(padded), None, None


class HartleyPool2d(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.h, self.w = _pair(pool_size)

    def forward(self, x):
        return SpectralPoolingFunction.apply(x, self.h, self.w)


class HartleyBottleneck(nn.Module):
    """ResNet bottleneck with the strided 3x3 replaced by spectral pooling."""

    expansion = 4

    def __init__(self, inplanes, planes, pool=None, downsample=None):
        super().__init__()
        self.pool = pool
        self.downsample = downsample

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        if self.pool is not None:
            out = self.pool(out)

        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class HartleyNet(nn.Module):
    def __init__(self, block=HartleyBottleneck, layers=(3, 4, 6, 3), num_classes=1):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.hsp = HartleyPool2d(pool_size=56)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], pool_size=28)
        self.layer3 = self._make_layer(block, 256, layers[2], pool_size=14)
        self.layer4 = self._make_layer(block, 512, layers[3], pool_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, pool_size=None):
        pool = HartleyPool2d(pool_size) if pool_size is not None else None
        downsample = None

        if self.inplanes != planes * block.expansion or pool_size is not None:
            skip_layers = []
            if self.inplanes != planes * block.expansion:
                skip_layers.extend([
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                ])
            if pool_size is not None:
                skip_layers.append(HartleyPool2d(pool_size))
            if skip_layers:
                downsample = nn.Sequential(*skip_layers)

        layers = [block(self.inplanes, planes, pool=pool, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.hsp(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def hartleynet_builder(pretrained_backbone=False):
    """Return an uninstantiated HartleyNet ready for ``load_state_dict``.

    During training the ImageNet ResNet-50 weights were used to warm-start
    the compatible convolutional layers. For inference the fine-tuned
    state_dict is loaded over the whole graph, so the ImageNet download
    is skipped by default.
    """
    model = HartleyNet()

    if pretrained_backbone:
        pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        pretrained_dict = pretrained.state_dict()
        model_dict = model.state_dict()
        filtered = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(filtered)
        model.load_state_dict(model_dict)

    return model
