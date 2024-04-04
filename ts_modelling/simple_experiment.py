import os
import time
import random

import numpy as np
import torch

from PatchTST.PatchTST_supervised.exp.exp_basic import Exp_Basic
# from PatchTST.PatchTST_supervised.data_provider.data_factory import data_provider
from simple_data_provider import SimpleDataProvider
from PatchTST.PatchTST_supervised.models import PatchTST
from PatchTST.PatchTST_supervised.utils.tools import EarlyStopping
from PatchTST.PatchTST_supervised.layers.PatchTST_backbone import Flatten_Head
from model_components import PretrainHead

from torch import optim
from torch import nn


class SimpleExp(Exp_Basic):
    """Trying to build a minimised experiment class for our needs
    """

    def __init__(self, args):
        super(SimpleExp, self).__init__(args)  # sets the device (GPU/CPU)

    def _get_data(self, flag):
        # data_set, data_loader = data_provider(self.args, flag)  # todo: make own simpler dataloader, w/o freqenc etc
        data_provider = SimpleDataProvider(self.args, flag)
        data_set, data_loader = data_provider.get_dataset_data_loader()
        return data_set, data_loader

    def _select_optimizer(self):  # function that is not required right now but if we want to add flexibility
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim

    def _select_criterion(self):
        losses = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'L1': nn.L1Loss(),
        }
        criterion = losses[self.args.loss]
        return criterion

    def _select_lr_scheduler(self, optimiser, train_steps):  # allow for others
        lr_schedulers = {
            'one_cycle_lr': optim.lr_scheduler.OneCycleLR(
                optimizer=optimiser,
                steps_per_epoch=train_steps,
                pct_start=self.args.lr_pct_start or 0.1,
                epochs=self.args.train_epochs,
                max_lr=self.args.lr
            )
        }
        return lr_schedulers[self.args.lr_scheduler]

    def _build_model(self):  # allow for other models
        model_dict = {
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        self.num_patches = int((self.args.seq_len - self.args.patch_len) / self.args.stride + 1)

        if self.args.training_task == 'self_supervised':
            model = self._swap_head(model=model, new_head='self_supervised')
            print('self_supervised head')
        elif self.args.training_task == 'supervised':
            print('supervised head')
        return model

    def infer_and_get_loss(self, batch_x, batch_y, model, criterion, task):
        if task == 'supervised':
            # get output, calculate loss as criterion of pred and true
            outputs = model(batch_x)
            f_dim = -1 if self.args.features == 'MS' else 0
            pred = outputs[:, -self.args.pred_len:, f_dim:]
            true = batch_y[:, -self.args.pred_len:, f_dim:]
            loss = criterion(pred, true)

        elif task == 'self_supervised':
            """if self supervised:
              draw random masked_indices of batch_x,
              predict using masked batch_x elementwise multiplied with NOT masked
              extract values at masked_indices from batch_x and output
              calculate loss as mse between output and unmasked_batch_x at indices"""

            # unmasked_batch_x = batch_x.clone()
            num_indices = int(self.args.mask_pct * self.num_patches)
            mask_indices_mtx = torch.zeros_like(batch_x, dtype=torch.bool)
            for b in range(batch_x.shape[0]):
                mask_start_indices = sorted(random.sample(range(self.num_patches), num_indices))
                mask_indices = []
                for index in mask_start_indices:
                    start = index * self.args.stride
                    end = start + self.args.patch_len
                    mask_indices.extend(range(start, end))
                batch_x[b, mask_indices, :] = 0  # todo: not ready for mv data
                mask_indices_mtx[b, mask_indices, :] = True

            outputs = model(batch_x * ~mask_indices_mtx)
            masked_outputs = outputs * mask_indices_mtx
            trues = batch_x * mask_indices_mtx

            loss = criterion(masked_outputs, trues)

        return loss

    def swap_head(self, new_head):
        """Changes head on model to type corresponding to new_head
        """

        self.model = self._swap_head(model=self.model, new_head=new_head)

    def _swap_head(self, model, new_head):
        if new_head == 'supervised':
            model.model.head = Flatten_Head(
                individual=self.args.individual,
                n_vars=self.args.enc_in,
                nf=self.args.d_model*(self.num_patches+1),
                target_window=self.args.pred_len,
                head_dropout=self.args.head_dropout
            )

        elif new_head == 'self_supervised':
            model.model.head = PretrainHead(
                d_model=self.args.d_model,
                patch_len=self.args.patch_len,
                dropout=self.args.dropout
            )

        return model

    def freeze_backbone(self, freeze=True):
        for param in self.model.model.backbone.parameters():
            param.requires_grad = not freeze

    def train(self, n_epochs=None):
        if n_epochs is None:
            n_epochs = self.args.train_epochs

        if self.args.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'Trainable parameters:\t{trainable_params}')
            print(f'Total parameters:\t\t{total_params}')

        tot_time_start = time.time()

        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        save_path = os.path.join(self.args.checkpoints, self.args.model_name, self.args.training_task)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=self.args.verbose,
        )

        optimiser = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_lr_scheduler(optimiser, train_steps)

        for epoch in range(n_epochs):
            epoch_time_start = time.time()
            train_loss = []

            self.model.train()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimiser.zero_grad()
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)

                # supervised training outputs = [bs x nvars x pred_len] forward prediction
                # self supervised outputs = [bs x nvars x seq_len]

                # infer_and_get_loss performs the training task (prediction/self_supervised) and returns the batch loss
                loss = self.infer_and_get_loss(
                    batch_x=batch_x,
                    batch_y=batch_y,
                    model=self.model,
                    criterion=criterion,
                    task=self.args.training_task,
                )

                train_loss.append(loss.item())

                if (i + 1) % self.args.batch_log_interval == 0:
                    print(f'train batch: {i + 1} | Loss: {loss.item():.7f}')

                loss.backward()
                optimiser.step()
                scheduler.step()

            train_loss = np.average(train_loss)
            val_loss = self.validate(val_loader, criterion)
            test_loss = self.validate(test_loader, criterion)

            if (epoch + 1) % self.args.epoch_log_interval == 0:
                print(
                    f'epoch: {epoch + 1}, '
                    f'train loss: {train_loss:.5}, '
                    f'validation loss: {val_loss:.5}, '
                    f'test loss: {test_loss:.5}'
                )
                print(
                    f'epoch time: {time.time() - epoch_time_start}'
                )
            early_stopping(val_loss=val_loss, model=self.model, path=save_path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            if self.args.verbose:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        self.model.load_state_dict(torch.load(os.path.join(save_path, 'checkpoint.pth')))

        total_training_time = time.time() - tot_time_start
        minutes = total_training_time // 60
        seconds = total_training_time % 60

        print(f'Total training time: {int(minutes)} minutes {seconds} seconds')
        return self.model

    def validate(self, val_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                loss = self.infer_and_get_loss(
                    batch_x=batch_x,
                    batch_y=batch_y,
                    model=self.model,
                    criterion=criterion,
                    task=self.args.training_task
                )

                total_loss.append(loss)

        self.model.train()
        return np.average(total_loss)

    def pretrain_model(self, n_epochs=None, task='self_supervised'):
        """If needed attaches pretrain head and trains on self_supervised setting"""
        if n_epochs is None:
            n_epochs = self.args.pretrain_epochs
        if not isinstance(self.model.model.head, PretrainHead):
            self.swap_head(new_head=task)
        self.train(n_epochs=n_epochs)

    def train_predict_head(self, n_epochs=None, task='supervised'):
        """if needed attaches prediction head and trains only head with frozen backbone"""
        if n_epochs is None:
            n_epochs = self.args.train_head_epochs
        if not isinstance(self.model.model.head, Flatten_Head):
            self.swap_head(new_head=task)
        self.args.training_task = 'supervised'
        self.freeze_backbone(freeze=True)
        self.train(n_epochs=n_epochs)

    def finetune_model(self, n_epochs=None):
        """unfreezes all parameters and trains"""
        if n_epochs is None:
            n_epochs = self.args.finetune_epochs
        self.freeze_backbone(freeze=False)
        self.train(n_epochs=n_epochs)

    def test(self, data_path=None):
        """
        data can be path to dataset or numpy array. If unspecified tests on test chunk of training data
        returns average test loss, saves trues and predictions to folder input_true_pred
        """

        pass

    def predict(self):
        pass

