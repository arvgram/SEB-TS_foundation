import os

from PatchTST.PatchTST_supervised.exp.exp_basic import Exp_Basic
from PatchTST.PatchTST_supervised.data_provider.data_factory import data_provider
from PatchTST.PatchTST_supervised.models.PatchTST import Model as PatchTST
from torch import optim
from torch import nn

from PatchTST.PatchTST_supervised.utils.tools import EarlyStopping


class SimpleExp(Exp_Basic):
    """Trying to build a minimised experiment class for our needs
    """
    def __init__(self, args):
        super(SimpleExp, self).__init__(args)  # sets the device (GPU/CPU)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)  # todo: make own simpler dataloader, w/o freqenc etc
        return data_set, data_loader

    def _select_optimizer(self):  # function that is not required right now but if we want to add flexibility
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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
                pct_start=self.args.lr_pct_start or 0.3,
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
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        save_path = os.path.join(self.args.checkpoints, self.args.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_steps = len(train_loader)
        early_stop = EarlyStopping(
            patience=self.args.patience,
            verbose=self.args.verbose,
        )

        optimiser = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_lr_scheduler(optimiser, train_steps)

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimiser.zero_grad()
                batch_x, batch_y = batch_x.float.to(self.device), batch_y.float.to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features is 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())







    def predict(self):

    def evaluate(self):


