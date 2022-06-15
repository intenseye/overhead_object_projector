import torch.nn as nn

CONSTANT_STD = 0.01


def init_with_normal(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=CONSTANT_STD)


class RegressionModel(nn.Module):
    def __init__(self, init_w_normal=False, projection_axis=None):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(4, 16)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(16, 64)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(64, 16)
        self.relu5 = nn.ReLU()
        if projection_axis == 'both':
            self.linear6 = nn.Linear(16, 2)
        else:
            self.linear6 = nn.Linear(16, 1)

        if init_w_normal:
            self.init_weights()

    def forward(self, x):

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.linear6(x)

        return x

    def init_weights(self):
        init_with_normal(self.modules())

