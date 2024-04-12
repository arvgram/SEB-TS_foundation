from torch import nn


class Model(nn.Module):
    """A naive predictor that repeats the same pattern forward"""
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        normalized_input = x - x[:, :1, :]
        last_values = x[:, -1:, :]
        repeated = normalized_input.repeat(1, int(self.pred_len/self.seq_len)+1, 1)
        output = last_values + repeated

        return output[:, :self.pred_len, :]