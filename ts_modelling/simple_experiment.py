import os
import time

import numpy as np
import torch
from torch.optim import Adam

from PatchTST.PatchTST_supervised.exp.exp_basic import Exp_Basic
from PatchTST.PatchTST_supervised.data_provider.data_factory import data_provider
from PatchTST.PatchTST_supervised.models.PatchTST import Model as PatchTST
from torch import load
from torch import optim
from torch import nn

from PatchTST.PatchTST_supervised.utils.tools import EarlyStopping


class SimpleExp(Exp_Basic):
    """Trying to build a minimised experiment class for our needs
    """

    def __init__(self, args):
        print('this is a changer version')
        super(SimpleExp, self).__init__(args)  # sets the device (GPU/CPU)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)  # todo: make own simpler dataloader, w/o freqenc etc
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
            'one_cycle_lr': optim.OneCycleLR(
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
        model = model_dict[self.args.model]
        return model

    def train(self):
        tot_time_start = time.time()

        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        save_path = os.path.join(self.args.checkpoints, self.args.model_name)
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

        for epoch in range(self.args.train_epochs):
            epoch_time_start = time.time()
            train_loss = []

            self.model.train()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimiser.zero_grad()
                batch_x, batch_y = batch_x.float.to(self.device), batch_y.float.to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
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
                    f'train loss: {train_loss:.7}, '
                    f'validation loss: {val_loss:}, '
                    f'test loss: {test_loss}'
                )
                print(
                    f'epoch time: {epoch_time_start - time.time()}'
                )
            early_stopping(val_loss=val_loss, model=self.model, path=save_path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            if self.args.verbose:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model = self.model.load_state_dict(load(os.join(save_path, 'checkpoint.pth')))

        total_training_time = time.time() - tot_time_start
        minutes = total_training_time // 60
        seconds = total_training_time % 60

        print(f'Total training time: {minutes} minutes {seconds} seconds')
        return best_model

    def validate(self, val_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss)

        self.model.train()
        return np.average(total_loss)

    def predict(self):
        pass
