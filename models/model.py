import torch
import torch.nn as nn
# import sys
# import os.path as osp
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from models import FPN, GhostNet


class Model(nn.Module):
    def __init__(self, freeze_bn=False, checkpoint=None):
        super(Model, self).__init__()
        self.backbone = GhostNet()
        self.fpn = FPN([self.backbone.low_outc, self.backbone.high_outc])

        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 10))

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self._init_weight()

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        high_level_feat, low_level_feat = self.backbone(input)
        x = self.fpn([low_level_feat, high_level_feat])
        x = self.last_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for module in [self.last_conv]:
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
