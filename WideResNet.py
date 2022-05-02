from abc import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as Spectral_Norm

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, use_sn=False):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(last_dim, num_classes)
        if use_sn:
            self.linear = Spectral_Norm(self.linear)

    @abstractmethod
    def penultimate(self, inputs):
        pass

    def forward(self, inputs, penultimate=False):
        _aux = {}
        _return_aux = False

        features = self.penultimate(inputs)
        output = self.linear(features)

        if penultimate:
            _return_aux = True
            _aux['penultimate'] = features

        if _return_aux:
            return output, _aux

        return output

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, use_sn=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        if use_sn:
            self.conv1 = Spectral_Norm(self.conv1)
            self.conv2 = Spectral_Norm(self.conv2)
            if isinstance(self.convShortcut, nn.Conv2d):
                self.convShortcut = Spectral_Norm(self.convShortcut)
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, use_sn):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, use_sn)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, use_sn):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, 
                                i == 0 and stride or 1, 
                                use_sn=use_sn))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(BaseModel):
    def __init__(self, n_classes=10, depth=34, widen_factor=1, use_sn=False):
        last_dim = 64 * widen_factor
        super(WideResNet, self).__init__(last_dim, n_classes, use_sn)
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        self.n_classes = n_classes

        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        if use_sn:
            self.conv1 = Spectral_Norm(self.conv1)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, use_sn=use_sn)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, use_sn=use_sn)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, use_sn=use_sn)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.last_dim = nChannels[3]
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def penultimate(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out
        
