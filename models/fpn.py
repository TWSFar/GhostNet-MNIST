import math
import torch.nn as nn
from models.selayer import SELayer


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel=256):
        super(FPN, self).__init__()
        C4_size, C5_size = in_channels

        # upsample C5 to get P5 from the FPN paper
        self.P5 = nn.Conv2d(C5_size, out_channel, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

        # add P5 elementwise to C4
        self.P4 = nn.Conv2d(C4_size, out_channel, kernel_size=1, stride=1, padding=0)
        self.selayer = SELayer(out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        C4, C5 = inputs

        P5_x = self.P5(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P4_x = self.P4(C4)
        P4_x = self.selayer(P4_x)
        res = P5_upsampled_x + P4_x

        return res
