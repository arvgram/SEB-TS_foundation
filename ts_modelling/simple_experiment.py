from PatchTST.PatchTST_supervised.exp.exp_basic import Exp_Basic
from PatchTST.PatchTST_supervised.data_provider.data_factory import data_provider

class SimpleExp(Exp_Basic):
    """Trying to build a minimised experiment class for our needs
    """
    def __init__(self, args):
        self.optimiser = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        super(SimpleExp, self).__init__(args)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def train(self):

    def predict(self):

    def evaluate(self):


