class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count