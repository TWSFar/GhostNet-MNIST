import torch
import torch.nn as nn


class CrossEntropyLoss(object):
    def __init__(self, weight=None, ignore_index=-1):
        self.ignore_index = ignore_index
        if weight is not None:
            self.weight = torch.tensor(weight, dtype=torch.float)
        else:
            self.weight = weight

    def __call__(self, logit, target):
        device = logit.device
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean').to(device)

        loss = criterion(logit, target.long())

        return loss


if __name__=='__main__':
    loss = CrossEntropyLoss()
    pred = torch.rand(2, 10)
    label = torch.tensor([0, 9])
    l = loss(pred, label)
    pass