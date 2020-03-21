import time
import logging
import numpy as np
from config import opt
from datasets.mnist import Mnist
from models.model import Model
from models.loss import CrossEntropyLoss
from utils import Timer, AverageMeter, Saver

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 


class Trainer(object):
    def __init__(self):
        # Configs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.saver = Saver(opt)
        self.logger = self.saver.log_info()
        self.best_pred = 0.0

        # Define model
        self.model = Model().to(self.device)

        # Datasets dataloader
        self.train_dataset = Mnist(data_dir=opt.data_dir, Train=True)
        self.val_dataset = Mnist(data_dir=opt.data_dir, Train=False)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.workers,
                                       shuffle=True,
                                       drop_last=False)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=opt.batch_size,
                                     num_workers=opt.workers,
                                     shuffle=False,
                                     drop_last=False)
        self.nbatch_train = len(self.train_loader)
        self.nbatch_val = len(self.val_loader)

        # Define Loss
        self.loss = CrossEntropyLoss()

        # Define Optimizer and Scheduler
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=opt.lr,
            momentum=opt.momentum,
            weight_decay=opt.decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[round(opt.epochs * x) for x in opt.steps],
            gamma=opt.gamma)

        # Time
        self.loss_hist = collections.deque(maxlen=500)
        self.timer = Timer(opt.epochs, self.nbatch_train, self.nbatch_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def train(self, epoch):
        self.model.train()
        last_time = time.time()
        for iter_num, imgs, labels in enumerate(tqdm(self.train_dataset)):
            imgs.to(self.device)
            labels.to(self.device)

            # Forward
            outputs = model(imgs)

            # Backward
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_hist.append(float(loss))

            # Visualization
            global_step = iter_num + self.nbatch_train * epoch + 1
            batch_time = time.time() - last_time
            last_time = time.time()
            self.step_time.append(batch_time)
            eta = self.timer.eta(global_step, batch_time)
            if global_step % opt.print_freq == 0:
                printline = ('Epoch: [{}][{}/{}] '
                             'lr: 1x:{:1.5f}, '
                             'eta: {}, time: {:1.1f}, '
                             'Loss: {:1.4f} '.format(
                                epoch, iter_num+1, self.nbatch_train,
                                self.optimizer.param_groups[0]['lr'],
                                eta, np.sum(self.step_time),
                                np.mean(self.loss_hist)))
                self.logger.info(printline)

        self.scheduler.step()

    def val(self, epoch):
        # Validate
        loss_meter = AverageMeter()
        correct_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_dataset):
                imgs.to(self.device)
                labels.to(self.device)

                outputs = self.model(imgs)

                loss = self.loss(outputs, labels)
                loss_meter.update(loss.item(), len(imgs))

                preds = torch.argmax(outputs, dim=1)
                correct = preds.eq(labels).sum().item()
                correct_meter.update(correct, 1)

        accuracy = correct_meter.sum / len(self.val_dataset)
        self.logger.info('Epoch: {}, accuracy: {}, average loss: {}, previous best'.format(
                            epoch, accuracy, loss, self.best_pred))

        return accuracy


def main():
    trainer = Trainer()
    start_time = time.time()
    for epoch in opt.epochs:
        # Train
        trainer.train()

        # Val
        val_time = time.time()
        accuracy = trainer.val()
        trainer.timer.set_val_eta(epoch, val_time-time.time())

        is_best = accuracy > trainer.best_pred
        trainer.best_pred = max(accuracy, trainer.best_pred)
        if (epoch % 20 == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)

    all_time = trainer.timer.second2hour(time.time() - start_time)
    print("Train done!, Sum time: {}, Best result: {}".format(all_time, trainer.best_pred))

if __name__=='__main__':
    main()