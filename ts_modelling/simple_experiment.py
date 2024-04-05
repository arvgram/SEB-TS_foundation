import os
import time
import random

import numpy as np
import torch

from PatchTST.PatchTST_supervised.exp.exp_basic import Exp_Basic
from PatchTST.PatchTST_supervised.models import PatchTST
from PatchTST.PatchTST_supervised.utils.tools import EarlyStopping
from PatchTST.PatchTST_supervised.layers.PatchTST_backbone import Flatten_Head

from results_utils import write_to_metrics_csv
from simple_data_provider import SimpleDataProvider
from model_components import PretrainHead

from torch import optim
from torch import nn


class SimpleExp(Exp_Basic):
    """Trying to build a minimised experiment class for our needs
    """

    def __init__(self, args):
        super(SimpleExp, self).__init__(args)  # sets the device (GPU/CPU)
        self.save_path = os.path.join(
            self.args.checkpoints,
            self.args.model_name,
        )
        self.test_results_path = os.path.join(
            './test_results/',
            self.args.model_name
        )

    def _get_data(self, flag):
        # data_set, data_loader = data_provider(self.args, flag)  # todo: make own simpler dataloader, w/o freqenc etc
        self.data_provider = SimpleDataProvider(self.args, flag)
        data_set, data_loader = self.data_provider.get_dataset_data_loader()
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

    def load_model(self, model_path=None):
        head = self.model.model.head
        if model_path is None:
            if os.path.exists(self.save_path + '/supervised'):
                if not isinstance(head, Flatten_Head):
                    self.swap_head('supervised')
                model_path = self.save_path + '/supervised'
            elif os.path.exists(self.save_path + '/self_supervised'):
                if not isinstance(head, PretrainHead):
                    self.swap_head('self_supervised')
                model_path = self.save_path + '/self_supervised'
            else:
                print('Please specify a valid model')
                return
            model_path += '/checkpoint.pth'

        self.model.load_state_dict(torch.load(model_path))

    def swap_head(self, new_head):
        """Changes head on model to type corresponding to new_head
        """
        self.model = self._swap_head(model=self.model, new_head=new_head)

    def _swap_head(self, model, new_head):
        if new_head == 'supervised':
            model.model.head = Flatten_Head(
                individual=self.args.individual,
                n_vars=self.args.enc_in,
                nf=self.args.d_model * (self.num_patches + 1),
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

        os.makedirs(self.save_path, exist_ok=True)

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
            early_stopping(val_loss=val_loss, model=self.model, path=self.save_path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            if self.args.verbose:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'checkpoint.pth')))

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
        """Test on test chunk of training data
        returns average test loss, saves trues and predictions to folder input_pred_true
        """
        old_data_path = self.args.data_path
        if data_path is not None:
            self.args.data_path = data_path

        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        inputs = []

        os.makedirs(self.test_results_path, exist_ok=True)

        self.model.eval()
        with (torch.no_grad()):
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                pred = outputs[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()  # removed .to(self.device)

                preds.append(pred)
                trues.append(true)
                inputs.append(batch_x.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)

        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        output_path = self.test_results_path + '/input_pred_true/'
        os.makedirs(output_path, exist_ok=True)

        np.save(output_path + 'input.npy', inputs)
        np.save(output_path + 'pred.npy', preds)
        np.save(output_path + 'true.npy', trues)

        write_to_metrics_csv(
            preds=preds,
            trues=trues,
            model_name=self.args.model_name,
            dataset=self.args.data_path,
            folder_path=self.test_results_path
        )

        self.args.data_path = old_data_path

    def plot_preds(self, nbr_plots=3, show=True):
        import numpy as np
        from matplotlib import pyplot as plt

        preds = np.load(self.test_results_path + '/input_pred_true/pred.npy')
        trues = np.load(self.test_results_path + '/input_pred_true/true.npy')
        inputs = np.load(self.test_results_path + '/input_pred_true/input.npy')

        plot_path = os.path.join(self.test_results_path, 'plots/')
        os.makedirs(plot_path, exist_ok=True)

        interval = trues.shape[0] // nbr_plots
        for i in range(nbr_plots):
            for j, col in zip(range(preds.shape[2]), self.data_provider.get_cols()):
                idx = i * interval
                print(f'channel {j}: {col}, position: {idx}')

                y = trues[idx, :, j]
                yhat = preds[idx, :, j]
                x = inputs[idx, :, j]

                plt.figure()
                plt.plot(x, label='input')
                plt.plot(range(len(x), len(x) + len(y)), y, label='true', alpha=0.5)
                plt.plot(range(len(x), len(x) + len(y)), yhat, label='pred')
                plt.title(f'Predictions and true values for variable {col}')
                plt.legend()
                plt.savefig(os.path.join(plot_path, f'channel-{col}_batch-{j}.pdf'), format='pdf')
                if show:
                    plt.show()

    def test_on_new_data(self, data_path):
        """use this to test on dataset that was not in training"""
        old_path = self.args.data_path

        self.args.data_path = old_path
        pass

    def predict(self):
        pass
