import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys





import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.pool  = nn.MaxPool2d(2, 2)
        self.up    = nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True)
        self.down  = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, inplane=None,planes=64, blocks=layers[0])
        self.layer1_0 = self._make_layer(block, inplane=None,planes=64,blocks= layers[0])
        self.layer1_1 = self._make_layer(block, inplane= None, planes=128, blocks=layers[0])
        self.layer1_2 = self._make_layer(block, inplane=192, planes=64, blocks=layers[0])
        print(self.layer1_2)
            
        # self.layer2 = self._make_layer(block, inplane= None, planes=128, blocks=layers[1], stride=2,
        #                                dilate=replace_stride_with_dilation[0]) 
        # self.conv2_0 = self._make_layer(block, nb_filter[1],  nb_filter[2], num_blocks[1])
        # self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        # self.conv0_2 = self._make_layer(block, nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.layer2_0 = self._make_layer(block, inplane=128,planes=256,blocks= layers[0])
        self.layer2_1 = self._make_layer(block, inplane=448,planes=128,blocks= layers[0])
        self.layer2_2 = self._make_layer(block, inplane=256,planes=128,blocks= layers[0])
        print(self.layer2_2)
        
        # self.layer3 = self._make_layer(block, inplane= None, planes=256, blocks=layers[2], stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        
        self.layer3_0 = self._make_layer(block, inplane=256,planes=512,blocks= layers[0])
        self.layer3_1 = self._make_layer(block, inplane=896,planes=256,blocks= layers[0])
        self.layer3_2 = self._make_layer(block, inplane=640,planes=128,blocks= layers[0])
        self.layer3_3 = self._make_layer(block, inplane=384,planes=256,blocks= layers[0])
        print(self.layer3_3)
        
        
        # self.layer4 = self._make_layer(block, inplane= None, planes=512, blocks=layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        
        self.layer4_0 = self._make_layer(block, inplane=512,planes=1024,blocks= layers[0])
        self.layer4_1 = self._make_layer(block, inplane=1792,planes=512,blocks= layers[0])
        self.layer4_2 = self._make_layer(block, inplane=1152,planes=256,blocks= layers[0])
        self.layer4_3 = self._make_layer(block, inplane=896,planes=128,blocks= layers[0])
        self.layer4_4 = self._make_layer(block, inplane=640,planes=512,blocks= layers[0])
        print(self.layer4_4)
        
        # print(self.layer4)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, inplane, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if inplane is not None:
            self.inplanes = inplane
        
        if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
      
        

        # x = self.layer1(x)
        # print(x.shape)
        x1 = self.layer1_0(x)
        # print("370",x1.shape)
        x2 = self.layer1_1(self.pool(x1))
        # print("372:",x2.shape)
        # print("374",x1.shape)
        # print("374",self.up(x2).shape)
        x2_1 = torch.cat([x1, self.up(x2)], 1)
        # print(x2_1.shape)
        x3 = self.layer1_2(x2_1)
        
        # c1 = self.layer2(x3)
        # x2_0 = self.conv2_0(self.pool(x1_0))
        # x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0),self.down(x0_1)], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        layer2_0 = self.layer2_0(self.pool(x2))
        layer2_1 = self.layer2_1(torch.cat([x2, self.up(layer2_0),self.down(x3)], 1))
        layer2_2 = self.layer2_2(torch.cat([x1, x3, self.up(layer2_1)], 1))
        
        
        
        layer3_0 = self.layer3_0(self.pool(layer2_0))
        layer3_1 = self.layer3_1(torch.cat([layer2_0, self.up(layer3_0),self.down(layer2_1)], 1))
        layer3_2 = self.layer3_2(torch.cat([x2, layer2_1, self.up(layer3_1),self.down(layer2_2)], 1))
        layer3_3 = self.layer3_3(torch.cat([x1, x3, layer2_2, self.up(layer3_2)], 1))


        # print(c1.shape)
        # c2 = self.layer3(layer2_3)
        # print(c2.shape)
        # c3 = self.layer4(layer3_3)
        
        layer4_0 = self.layer4_0(self.pool(layer3_0))
        layer4_1 = self.layer4_1(torch.cat([layer3_0, self.up(layer4_0),self.down(layer3_1)], 1))
        layer4_2 = self.layer4_2(torch.cat([layer2_0, layer3_1, self.up(layer4_1),self.down(layer3_2)], 1))
        layer4_3 = self.layer4_3(torch.cat([x2, layer2_1, layer3_2, self.up(layer4_2),self.down(layer3_3)], 1))
        layer4_4 = self.layer4_4(torch.cat([x1, x3, layer2_2, layer3_3, self.up(layer4_3)], 1))
        
        
        # print(c3.shape)
        # 在需要停止程序的地方调用sys.exit()
        # sys.exit()

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return layer2_2, layer3_3, layer4_4

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    pretrained=False
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)