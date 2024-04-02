from torch import nn


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patch x patch_len]
        x = x.flatten(start_dim=2)

        # old:
        # x = x.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x patch_len]
        # x = x.flatten(start_dim=1, end_dim=3)
        return x
