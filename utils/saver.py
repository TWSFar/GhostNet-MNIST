import os
import shutil
import os.path as osp

import logging
import torch
from tensorboardX import SummaryWriter


class Saver(object):
    def __init__(self, opt):
        self.opt = opt
        self.directory = osp.join(opt.work_dir)
        self.logfile = osp.join(self.directory, 'train.log')
        if not osp.exists(self.directory):
            os.makedirs(self.directory)
        logging.basicConfig(
                    format='[%(asctime)s %(levelname)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)
        f_handler = logging.FileHandler(self.logfile, mode='w')
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(f_handler)
        for key, val in self.opt._state_dict().items():
            self.logger.info(key + ': ' + str(val))
        self.writer = SummaryWriter(log_dir=self.directory)

    def save_checkpoint(self, state, is_best, filename='last.pth'):
        ''' Saver checkpoint to disk '''
        filename = os.path.join(self.directory, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth'))
