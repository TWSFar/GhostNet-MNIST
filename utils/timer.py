import time
import collections
import numpy as np


class Timer(object):
    def __init__(self, epoch, train_bn, val_bn, val_freq=1):
        self.epoch = epoch
        self.train_step = train_bn * epoch
        # use average step time to calculate eta time
        self.batch_time = collections.deque(maxlen=500)
        self.val_step = val_bn * epoch
        self.val_freq = val_freq
        self.val_eta_time = -1

    def eta(self, cur_step, batch_time):
        self.batch_time.append(batch_time)
        if self.val_eta_time == -1:
            self.val_eta_time = batch_time * self.val_step / self.val_freq
        eta = (self.train_step - cur_step) * np.mean(self.batch_time) + self.val_eta_time
        return self.second2hour(eta)

    def set_val_eta(self, cur_epoch, val_time):
        self.val_eta_time = (self.epoch - cur_epoch) / self.val_freq * val_time

    def second2hour(self, s):
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return ("%02d:%02d:%02d" % (h, m, s))
