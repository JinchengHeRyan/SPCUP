import torch.nn as nn


class linear_det(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(linear_det, self).__init__()
        self.linear = nn.Linear(in_features=in_feature, out_features=out_feature)

    def forward(self, x):
        x = self.linear(x)
        return x
