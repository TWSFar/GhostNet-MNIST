import torch
import torch.nn as nn
# import sys
# import os.path as osp
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from models import ASPP, FPN, GhostNet


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.backbone = GhostNet()
        self.aspp = ASPP(
            opt.backbone,
            opt.output_stride,
            self.backbone.high_outc,
            nn.BatchNorm2d)
        self.link_conv = nn.Sequential(nn.Conv2d(
            self.backbone.low_outc, 128, kernel_size=1, stride=1, padding=0, bias=False))
        self.last_conv = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 10))

        self._init_weight()
        if opt.freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        # low_level_feat = self.link_conv(low_level_feat)
        # x = torch.cat((x, low_level_feat), dim=1)
        x = self.aspp(x)
        x = self.last_conv(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for module in [self.link_conv, self.last_conv]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


if __name__ == "__main__":
    model = Model(backbone='mobilenetv2', output_stride=16)
    model.eval()
    input = torch.rand(5, 3, 640, 480)
    output = model(input)
    pass
