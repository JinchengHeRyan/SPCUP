import numpy as np
import pdb
import logging
import os
import torch
from utils.average import AverageVal
from utils.utils import acc
from utils.loss import Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import librosa


class Trainer:
    def __init__(
        self,
        chkpt_dir,
        model,
        optimizer,
        criterion,
        lr_scheduler,
        logger,
        epochs,
        trainloader,
        testloader=None,
        resume_path=None,
        print_freq=10,
    ):
        """
        Initialize all the components, and resume model if resume_path is not None.
        """
        super(Trainer, self).__init__()
        self.chkpt_dir = chkpt_dir
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.start_epoch = 0
        self.epochs = epochs
        self.trainloader = trainloader
        self.testloader = testloader
        self.print_freq = print_freq
        if resume_path is not None:
            self._resume_checkpoint(resume_path)

    def _save_checkpoint(self, epoch, isbest=False):
        """
        Saving checkpoints
        -----------------
        Params:
            epoch: current epoch to be saved.
            isbest: if True, also save current epoch to `model_best.pth`
        -----------------
        return:
            None
        """
        state = {
            "epoch": epoch,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        save_path = os.path.join(self.chkpt_dir, f"chkpt_{epoch:03d}.pth")
        torch.save(state, save_path)
        self.logger.info(f"Saving checkpoint: {save_path} ...")

        # save the best model
        if isbest:
            best_path = os.path.join(self.chkpt_dir, "model_best.pth")
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: {best_path} ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        -------------------
        Params:
            resume_path: path to resume the model.
        -------------------
        return:
            None
        """
        assert os.path.isfile(resume_path), f"No such file {resume_path}"
        chkpt = torch.load(resume_path)

        self.logger.info(f"Resume from checkpoint: {resume_path}")
        self.start_epoch = chkpt["epoch"] + 1
        self.model.module.load_state_dict(chkpt["model"])
        self.optimizer.load_state_dict(chkpt["optimizer"])
        self.lr_scheduler.load_state_dict(chkpt["lr_scheduler"])

    def _train_epoch(self, epoch):
        """
        Train for one epoch
        ------------
        Params:
            epoch: the current epoch number
        ------------
        return:
            loss.avg: The average loss for the current epoch
        """
        self.model.train()

        losses = AverageVal()
        accs = AverageVal()

        datatime = 0
        batchtime = 0
        tic = time.time()
        for i, (wav_utt, fbank, wav_spec, wav_phase, wav, target_method) in enumerate(
            self.trainloader
        ):
            datatime += time.time() - tic

            fbank = (
                fbank.view(fbank.shape[0], 1, fbank.shape[1], fbank.shape[2])
                .float()
                .cuda()
            )

            target_method = target_method.cuda()

            pred = self.model(fbank)

            loss = self.criterion(pred, target_method)
            losses.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            accuracy = acc(pred, target_method)
            accs.update(accuracy)

            batchtime += time.time() - tic
            tic = time.time()

            if (i + 1) % self.print_freq == 0:
                self.logger.info(
                    "[Train] Epoch:{} [{:03d}/{:03d}]({:.0f}%)\t"
                    "Time:{:.3f}/{:.3f}\tLoss:({:.4f}){:.4f}\tAcc:({:.4f}){:.4f}".format(
                        epoch,
                        i + 1,
                        len(self.trainloader),
                        (i + 1) / len(self.trainloader) * 100,
                        datatime,
                        batchtime,
                        losses.val,
                        losses.avg,
                        accs.val,
                        accs.avg,
                    )
                )
                datatime = 0
                batchtime = 0

        self.logger.info(
            "[Train Summary] Epoch:{}\t Loss:{:.4f}\t Acc:{:.4f}".format(
                epoch, losses.avg, accs.avg
            )
        )
        return losses.avg

    def _test_epoch(self, epoch):
        """
        Validate after training one epoch.
        Skipped in training process if self.testloader is None.
        ------------
        Params:
            epoch: the current epoch number
        ------------
        return:
            loss.avg: The average loss for the current epoch
        """
        self.model.eval()
        losses = AverageVal()
        accs = AverageVal()

        datatime = 0
        batchtime = 0
        tic = time.time()
        for i, (wav_utt, fbank, wav_spec, wav_phase, wav, target_method) in enumerate(
            self.testloader
        ):
            datatime += time.time() - tic

            with torch.no_grad():
                fbank = (
                    fbank.view(fbank.shape[0], 1, fbank.shape[1], fbank.shape[2])
                    .float()
                    .cuda()
                )
                target_method = target_method.cuda()
                pred = self.model(fbank)

            loss = self.criterion(pred, target_method)
            losses.update(loss.item())

            # Compute accuracy
            accuracy = acc(pred, target_method)
            accs.update(accuracy)

            batchtime += time.time() - tic
            tic = time.time()

            if (i + 1) % self.print_freq == 0:
                self.logger.info(
                    "[Test] Epoch:{} [{:03d}/{:03d}]({:.0f}%)\t"
                    "Time:{:.3f}/{:.3f}\tLoss:({:.4f}){:.4f}\tAcc:({:.4f}){:.4f}".format(
                        epoch,
                        i + 1,
                        len(self.testloader),
                        (i + 1) / len(self.testloader) * 100,
                        datatime,
                        batchtime,
                        losses.val,
                        losses.avg,
                        accs.val,
                        accs.avg,
                    )
                )
                datatime = 0
                batchtime = 0

        self.logger.info(
            "[Test Summary] Epoch:{}\t Loss:{:.4f}\t Acc:{:.4f}".format(
                epoch, losses.avg, accs.avg
            )
        )
        self.model.train()
        return losses.avg

    def train(self):
        """
        Train for N epochs.
        """
        min_loss = 100
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss = self._train_epoch(epoch)
            if self.testloader is not None:
                epoch_loss = self._test_epoch(epoch)

            # adjust the learning rate. !!! MIN for loss !!!
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(epoch_loss)
            else:
                self.lr_scheduler.step()

            # save the current epoch.
            # If min_loss > epoch_loss, also save as the best_model.
            isbest = min_loss > epoch_loss
            min_loss = min(min_loss, epoch_loss)
            self._save_checkpoint(epoch, isbest)
