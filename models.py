from typing import Any
import torch
import torch.nn as nn

CONSTANT_STD = 0.01  # standard deviation value used for normally-distributed weight initialization


def init_with_normal(modules: Any):
    """
    Initialize the weight values from the normal distribution

    Parameters
    ----------
    modules: Any
        Modules of the model
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=CONSTANT_STD)


def set_activation_function(activation: str = 'relu') -> Any:
    """
    Sets the activation function.

    Parameters
    ----------
    activation: str
        Activation tag

    Returns
    ----------
    activation_function: Any
        Activation function
    """
    activation_function = None
    if activation == 'relu':
        activation_function = nn.ReLU()
    elif activation == 'prelu':
        activation_function = nn.PReLU()
    elif activation == 'leaky_reLU':
        activation_function = nn.LeakyReLU()
    elif activation == 'gelu':
        activation_function = nn.GELU()
    elif activation == 'elu':
        activation_function = nn.ELU()
    elif activation == 'selu':
        activation_function = nn.SELU()
    return activation_function


class RegressionModel(nn.Module):
    """
    Regression Model base class
    """
    def __init__(self, activation: str, use_batch_norm: bool):
        """
        Initialize the base Regression Model class.

        Parameters
        ----------
        activation: str
            Activation tag
        use_batch_norm: bool
            Enables batch normalization
        """
        super(RegressionModel, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.activation_function = set_activation_function(activation)
        self.linear_bias = True
        if self.use_batch_norm is True:
            self.linear_bias = False

    def init_weights(self):
        """
        Initialize weights of the model.
        """
        init_with_normal(self.modules())


class RegressionModelXLarge(RegressionModel):
    """
    XLarge Regression Model class
    """
    def __init__(self, projection_axis: str = 'x', activation: str = 'relu', init_w_normal: bool = False,
                 use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the XLarge Regression Model class.

        Parameters
        ----------
        projection_axis: str
            The projection axis
        activation: str
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: bool
            Momentum value used in batch normalization
        """

        super(RegressionModelXLarge, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
            self.bn2 = nn.BatchNorm1d(64, momentum=batch_momentum)
            self.bn3 = nn.BatchNorm1d(256, momentum=batch_momentum)
            self.bn4 = nn.BatchNorm1d(1024, momentum=batch_momentum)
            self.bn5 = nn.BatchNorm1d(256, momentum=batch_momentum)
            self.bn6 = nn.BatchNorm1d(64, momentum=batch_momentum)
            self.bn7 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)
        self.linear2 = nn.Linear(16, 64, bias=self.linear_bias)
        self.linear3 = nn.Linear(64, 256, bias=self.linear_bias)
        self.linear4 = nn.Linear(256, 1024, bias=self.linear_bias)
        self.linear5 = nn.Linear(1024, 256, bias=self.linear_bias)
        self.linear6 = nn.Linear(256, 64, bias=self.linear_bias)
        self.linear7 = nn.Linear(64, 16, bias=self.linear_bias)

        if projection_axis == 'both':
            self.linear8 = nn.Linear(16, 2)
        else:
            self.linear8 = nn.Linear(16, 1)

        if init_w_normal is True:
            self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function of the model

        Parameters
        ----------
        x: torch.Tensor
            Input of the model

        Returns
        ----------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        if self.use_batch_norm is True:
            x = self.bn1(x)
        x = self.activation_function(x)

        x = self.linear2(x)
        if self.use_batch_norm is True:
            x = self.bn2(x)
        x = self.activation_function(x)

        x = self.linear3(x)
        if self.use_batch_norm is True:
            x = self.bn3(x)
        x = self.activation_function(x)

        x = self.linear4(x)
        if self.use_batch_norm is True:
            x = self.bn4(x)
        x = self.activation_function(x)

        x = self.linear5(x)
        if self.use_batch_norm is True:
            x = self.bn5(x)
        x = self.activation_function(x)

        x = self.linear6(x)
        if self.use_batch_norm is True:
            x = self.bn6(x)
        x = self.activation_function(x)

        x = self.linear7(x)
        if self.use_batch_norm is True:
            x = self.bn7(x)
        x = self.activation_function(x)

        x = self.linear8(x)

        return x


class RegressionModelLarge(RegressionModel):
    """
    Initialize the Large Regression Model class.

    Parameters
    ----------
    projection_axis: str
        The projection axis
    activation: str
        Activation tag
    init_w_normal: bool
        Enables the normal distribution based weight initialization
    use_batch_norm: bool
        Enables batch normalization
    batch_momentum: bool
        Momentum value used in batch normalization
    """

    def __init__(self, projection_axis: str = 'x', activation: str = 'relu', init_w_normal: bool = False,
                 use_batch_norm: bool = False, batch_momentum: float = 0.1):
        super(RegressionModelLarge, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
            self.bn2 = nn.BatchNorm1d(64, momentum=batch_momentum)
            self.bn3 = nn.BatchNorm1d(128, momentum=batch_momentum)
            self.bn4 = nn.BatchNorm1d(64, momentum=batch_momentum)
            self.bn5 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)
        self.linear2 = nn.Linear(16, 64, bias=self.linear_bias)
        self.linear3 = nn.Linear(64, 128, bias=self.linear_bias)
        self.linear4 = nn.Linear(128, 64, bias=self.linear_bias)
        self.linear5 = nn.Linear(64, 16, bias=self.linear_bias)

        if projection_axis == 'both':
            self.linear6 = nn.Linear(16, 2)
        else:
            self.linear6 = nn.Linear(16, 1)

        if init_w_normal is True:
            self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function of the model

        Parameters
        ----------
        x: torch.Tensor
            Input of the model

        Returns
        ----------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        if self.use_batch_norm is True:
            x = self.bn1(x)
        x = self.activation_function(x)

        x = self.linear2(x)
        if self.use_batch_norm is True:
            x = self.bn2(x)
        x = self.activation_function(x)

        x = self.linear3(x)
        if self.use_batch_norm is True:
            x = self.bn3(x)
        x = self.activation_function(x)

        x = self.linear4(x)
        if self.use_batch_norm is True:
            x = self.bn4(x)
        x = self.activation_function(x)

        x = self.linear5(x)
        if self.use_batch_norm is True:
            x = self.bn5(x)
        x = self.activation_function(x)

        x = self.linear6(x)

        return x


class RegressionModelMedium(RegressionModel):
    """
    Medium Regression Model class
    """
    def __init__(self, projection_axis: str = 'x', activation: str = 'relu', init_w_normal: bool = False,
                 use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the Medium Regression Model class.

        Parameters
        ----------
        projection_axis: str
            The projection axis
        activation: str
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: bool
            Momentum value used in batch normalization
        """

        super(RegressionModelMedium, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
            self.bn2 = nn.BatchNorm1d(64, momentum=batch_momentum)
            self.bn3 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)
        self.linear2 = nn.Linear(16, 64, bias=self.linear_bias)
        self.linear3 = nn.Linear(64, 16, bias=self.linear_bias)

        if projection_axis == 'both':
            self.linear4 = nn.Linear(16, 2)
        else:
            self.linear4 = nn.Linear(16, 1)

        if init_w_normal is True:
            self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function of the model

        Parameters
        ----------
        x: torch.Tensor
            Input of the model

        Returns
        ----------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        if self.use_batch_norm is True:
            x = self.bn1(x)
        x = self.activation_function(x)

        x = self.linear2(x)
        if self.use_batch_norm is True:
            x = self.bn2(x)
        x = self.activation_function(x)

        x = self.linear3(x)
        if self.use_batch_norm is True:
            x = self.bn3(x)
        x = self.activation_function(x)

        x = self.linear4(x)

        return x


class RegressionModelSmall(RegressionModel):
    """
    Initialize the Small Regression Model class.

    Parameters
    ----------
    projection_axis: str
        The projection axis
    activation: str
        Activation tag
    init_w_normal: bool
        Enables the normal distribution based weight initialization
    use_batch_norm: bool
        Enables batch normalization
    batch_momentum: bool
        Momentum value used in batch normalization
    """

    def __init__(self, projection_axis: str = 'x', activation: str = 'relu', init_w_normal: bool = False,
                 use_batch_norm: bool = False, batch_momentum: float = 0.1):
        super(RegressionModelSmall, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
            self.bn2 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)
        self.linear2 = nn.Linear(16, 16, bias=self.linear_bias)

        if projection_axis == 'both':
            self.linear3 = nn.Linear(16, 2)
        else:
            self.linear3 = nn.Linear(16, 1)

        if init_w_normal is True:
            self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function of the model

        Parameters
        ----------
        x: torch.Tensor
            Input of the model

        Returns
        ----------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        if self.use_batch_norm is True:
            x = self.bn1(x)
        x = self.activation_function(x)

        x = self.linear2(x)
        if self.use_batch_norm is True:
            x = self.bn2(x)
        x = self.activation_function(x)

        x = self.linear3(x)

        return x


class RegressionModelXSmall(RegressionModel):
    """
    XSmall Regression Model class
    """
    def __init__(self, projection_axis: str = 'x', activation: str = 'relu', init_w_normal: bool = False,
                 use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the XSmall Regression Model class.

        Parameters
        ----------
        projection_axis: str
            The projection axis
        activation: str
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: bool
            Momentum value used in batch normalization
        """

        super(RegressionModelXSmall, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)

        if projection_axis == 'both':
            self.linear2 = nn.Linear(16, 2)
        else:
            self.linear2 = nn.Linear(16, 1)
        if init_w_normal is True:
            self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function of the model

        Parameters
        ----------
        x: torch.Tensor
            Input of the model

        Returns
        ----------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        if self.use_batch_norm is True:
            x = self.bn1(x)
        x = self.activation_function(x)
        x = self.linear2(x)
        return x


class RegressionModelLinear(RegressionModel):
    """
    Initialize the Linear Regression Model class.

    Parameters
    ----------
    projection_axis: str
        The projection axis
    activation: str
        Activation tag
    init_w_normal: bool
        Enables the normal distribution based weight initialization
    use_batch_norm: bool
        Enables batch normalization
    batch_momentum: bool
        Momentum value used in batch normalization
    """

    def __init__(self, projection_axis: str = 'x', activation: str = 'relu', init_w_normal: bool = False,
                 use_batch_norm: bool = False, batch_momentum: float = 0.1):
        super(RegressionModelLinear, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if projection_axis == 'both':
            self.linear1 = nn.Linear(4, 2)
        else:
            self.linear1 = nn.Linear(4, 1)
        if init_w_normal is True:
            self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function of the model

        Parameters
        ----------
        x: torch.Tensor
            Input of the model

        Returns
        ----------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        return x
