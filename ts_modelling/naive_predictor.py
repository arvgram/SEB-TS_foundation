from torch import nn


class Model(nn.Module):
    """A naive predictor that repeats the last value"""
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        last_values = x[:, -1, :]

        # Repeat the last value pred_len times
        repeated_last_values = last_values.unsqueeze(1).repeat(1, self.pred_len, 1)

        return repeated_last_values
