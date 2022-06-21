import torch.nn as nn

CONSTANT_STD = 0.01


def init_with_normal(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=CONSTANT_STD)


class RegressionModelXLarge(nn.Module):
    def __init__(self, projection_axis='x', init_w_normal=False, use_batch_norm=False, batch_momentum=0.1):
        super(RegressionModelXLarge, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.linear1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(16, 64)
        self.bn2 = nn.BatchNorm1d(64, momentum=batch_momentum)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(64, 256)
        self.bn3 = nn.BatchNorm1d(128, momentum=batch_momentum)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(256, 1024)
        self.bn4 = nn.BatchNorm1d(128, momentum=batch_momentum)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(1024, 256)
        self.bn5 = nn.BatchNorm1d(128, momentum=batch_momentum)
        self.relu5 = nn.ReLU()

        self.linear6 = nn.Linear(256, 64)
        self.bn6 = nn.BatchNorm1d(64, momentum=batch_momentum)
        self.relu6 = nn.ReLU()

        self.linear7 = nn.Linear(64, 16)
        self.bn7 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu7 = nn.ReLU()

        if projection_axis == 'both':
            self.linear8 = nn.Linear(16, 2)
        else:
            self.linear8 = nn.Linear(16, 1)

        if init_w_normal:
            self.init_weights()

    def forward(self, x):

        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)

        x = self.linear3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = self.linear4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.relu4(x)

        x = self.linear5(x)
        if self.use_batch_norm:
            x = self.bn5(x)
        x = self.relu5(x)

        x = self.linear6(x)
        if self.use_batch_norm:
            x = self.bn6(x)
        x = self.relu6(x)

        x = self.linear7(x)
        if self.use_batch_norm:
            x = self.bn7(x)
        x = self.relu7(x)

        x = self.linear8(x)

        return x

    def init_weights(self):
        init_with_normal(self.modules())


class RegressionModelLarge(nn.Module):
    def __init__(self, projection_axis='x', init_w_normal=False, use_batch_norm=False, batch_momentum=0.1):
        super(RegressionModelLarge, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.linear1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(16, 64)
        self.bn2 = nn.BatchNorm1d(64, momentum=batch_momentum)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128, momentum=batch_momentum)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64, momentum=batch_momentum)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(64, 16)
        self.bn5 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu5 = nn.ReLU()

        if projection_axis == 'both':
            self.linear6 = nn.Linear(16, 2)
        else:
            self.linear6 = nn.Linear(16, 1)

        if init_w_normal:
            self.init_weights()

    def forward(self, x):

        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)

        x = self.linear3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = self.linear4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = self.relu4(x)

        x = self.linear5(x)
        if self.use_batch_norm:
            x = self.bn5(x)
        x = self.relu5(x)

        x = self.linear6(x)

        return x

    def init_weights(self):
        init_with_normal(self.modules())


class RegressionModelMedium(nn.Module):
    def __init__(self, projection_axis='x', init_w_normal=False, use_batch_norm=False, batch_momentum=0.1):
        super(RegressionModelMedium, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.linear1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(16, 64)
        self.bn2 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu3 = nn.ReLU()

        if projection_axis == 'both':
            self.linear4 = nn.Linear(16, 2)
        else:
            self.linear4 = nn.Linear(16, 1)

        if init_w_normal:
            self.init_weights()

    def forward(self, x):

        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)

        x = self.linear3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = self.linear4(x)

        return x

    def init_weights(self):
        init_with_normal(self.modules())


class RegressionModelSmall(nn.Module):
    def __init__(self, projection_axis='x', init_w_normal=False, use_batch_norm=False, batch_momentum=0.1):
        super(RegressionModelSmall, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.linear1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(16, 16)
        self.bn2 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu2 = nn.ReLU()

        if projection_axis == 'both':
            self.linear3 = nn.Linear(16, 2)
        else:
            self.linear3 = nn.Linear(16, 1)

        if init_w_normal:
            self.init_weights()

    def forward(self, x):

        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu2(x)

        x = self.linear3(x)

        return x

    def init_weights(self):
        init_with_normal(self.modules())


class RegressionModelXSmall(nn.Module):
    def __init__(self, projection_axis='x', init_w_normal=False, use_batch_norm=False, batch_momentum=0.1):
        super(RegressionModelXSmall, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.linear1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
        self.relu1 = nn.ReLU()

        if projection_axis == 'both':
            self.linear2 = nn.Linear(16, 2)
        else:
            self.linear2 = nn.Linear(16, 1)
        if init_w_normal:
            self.init_weights()

    def forward(self, x):
        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

    def init_weights(self):
        init_with_normal(self.modules())


