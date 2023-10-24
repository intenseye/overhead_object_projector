from typing import Iterator
from enum import Enum
import torch
import torch.nn as nn

CONSTANT_STD = 0.01  # standard deviation value used for normally-distributed weight initialization


def init_with_normal(modules: Iterator[nn.Module]):
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


class ActivationType(Enum):
    """
    Enumeration representing different activation types used in neural networks.

    RELU: Rectified Linear Unit (ReLU) activation function.
    PRELU: Parametric Rectified Linear Unit (PReLU) activation function.
    LEAKY_RELU: Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    GELU: Gaussian Error Linear Unit (GELU) activation function.
    ELU: Exponential Linear Unit (ELU) activation function.
    SELU: Scaled Exponential Linear Unit (SELU) activation function.
    """
    RELU = 'relu'
    PRELU = 'prelu'
    LEAKY_RELU = 'leaky_relu'
    GELU = 'gelu'
    ELU = 'elu'
    SELU = 'selu'


def set_activation_function(activation: ActivationType = ActivationType.RELU) -> nn.Module:
    """
    Sets the activation function.

    Parameters
    ----------
    activation: ActivationType, optional
        Activation type (default is ActivationType.RELU)

    Returns
    -------
    activation_function: nn.Module
        Activation function
    """
    activation_function = None
    if activation == ActivationType.RELU:
        activation_function = nn.ReLU()
    elif activation == ActivationType.PRELU:
        activation_function = nn.PReLU()
    elif activation == ActivationType.LEAKY_RELU:
        activation_function = nn.LeakyReLU()
    elif activation == ActivationType.GELU:
        activation_function = nn.GELU()
    elif activation == ActivationType.ELU:
        activation_function = nn.ELU()
    elif activation == ActivationType.SELU:
        activation_function = nn.SELU()
    return activation_function


class ProjectionAxis(Enum):
    """
    Enumeration representing different projection axes.
    x represents projection along the x-axis.
    y represents projection along the y-axis.
    both represents projection along both x and y axes.
    """
    x = 'x'
    y = 'y'
    both = 'both'


class OverProjNet(nn.Module):
    """
    OverProjNet base class
    """

    def __init__(self, activation: ActivationType, use_batch_norm: bool):
        """
        Initialize the base OverProjNet class.

        Parameters
        ----------
        activation: ActivationType
            Activation tag
        use_batch_norm: bool
            Enables batch normalization
        """
        super(OverProjNet, self).__init__()
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


class OverProjNetXL(OverProjNet):
    """
    XLarge OverProjNet class
    """

    def __init__(self, projection_axis: ProjectionAxis = ProjectionAxis.x, activation: ActivationType = ActivationType.RELU,
                 init_w_normal: bool = False, use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the XLarge OverProjNet class.

        Parameters
        ----------
        projection_axis: ProjectionAxis
            The projection axis
        activation: ActivationType
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: float
            Momentum value used in batch normalization
        """

        super(OverProjNetXL, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

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

        if projection_axis == ProjectionAxis.both:
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
        -------
        x: torch.Tensor
            Output of the model
        """

        for i in range(1, 8):
            x = getattr(self, f'linear{i}')(x)
            if self.use_batch_norm:
                x = getattr(self, f'bn{i}')(x)
            x = self.activation_function(x)

        x = self.linear8(x)
        return x


class OverProjNetL(OverProjNet):
    """
    Large OverProjNet class
    """

    def __init__(self, projection_axis: ProjectionAxis = ProjectionAxis.x, activation: ActivationType = ActivationType.RELU,
                 init_w_normal: bool = False, use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the Large OverProjNet class.

        Parameters
        ----------
        projection_axis: ProjectionAxis
            The projection axis
        activation: ActivationType
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: float
            Momentum value used in batch normalization
        """
        super(OverProjNetL, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

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

        if projection_axis == ProjectionAxis.both:
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
        -------
        x: torch.Tensor
            Output of the model
        """

        for i in range(1, 6):
            x = getattr(self, f'linear{i}')(x)
            if self.use_batch_norm:
                x = getattr(self, f'bn{i}')(x)
            x = self.activation_function(x)

        x = self.linear6(x)

        return x


class OverProjNetM(OverProjNet):
    """
    Medium OverProjNet class
    """

    def __init__(self, projection_axis: ProjectionAxis = ProjectionAxis.x, activation: ActivationType = ActivationType.RELU,
                 init_w_normal: bool = False, use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the Medium OverProjNet class.

        Parameters
        ----------
        projection_axis: ProjectionAxis
            The projection axis
        activation: ActivationType
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: float
            Momentum value used in batch normalization
        """

        super(OverProjNetM, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
            self.bn2 = nn.BatchNorm1d(64, momentum=batch_momentum)
            self.bn3 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)
        self.linear2 = nn.Linear(16, 64, bias=self.linear_bias)
        self.linear3 = nn.Linear(64, 16, bias=self.linear_bias)

        if projection_axis == ProjectionAxis.both:
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
        -------
        x: torch.Tensor
            Output of the model
        """

        for i in range(1, 4):
            x = getattr(self, f'linear{i}')(x)
            if self.use_batch_norm:
                x = getattr(self, f'bn{i}')(x)
            x = self.activation_function(x)

        x = self.linear4(x)

        return x


class OverProjNetS(OverProjNet):
    """
    Small OverProjNet class
    """

    def __init__(self, projection_axis: ProjectionAxis = ProjectionAxis.x, activation: ActivationType = ActivationType.RELU,
                 init_w_normal: bool = False, use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the Small OverProjNet class.

        Parameters
        ----------
        projection_axis: ProjectionAxis
            The projection axis
        activation: ActivationType
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: float
            Momentum value used in batch normalization
        """
        super(OverProjNetS, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)
            self.bn2 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)
        self.linear2 = nn.Linear(16, 16, bias=self.linear_bias)

        if projection_axis == ProjectionAxis.both:
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
        -------
        x: torch.Tensor
            Output of the model
        """

        for i in range(1, 3):
            x = getattr(self, f'linear{i}')(x)
            if self.use_batch_norm:
                x = getattr(self, f'bn{i}')(x)
            x = self.activation_function(x)

        x = self.linear3(x)

        return x


class OverProjNetXS(OverProjNet):
    """
    XSmall OverProjNet class
    """

    def __init__(self, projection_axis: ProjectionAxis = ProjectionAxis.x, activation: ActivationType = ActivationType.RELU,
                 init_w_normal: bool = False, use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the XSmall OverProjNet class.

        Parameters
        ----------
        projection_axis: ProjectionAxis
            The projection axis
        activation: ActivationType
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: float
            Momentum value used in batch normalization
        """

        super(OverProjNetXS, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if self.use_batch_norm is True:
            self.bn1 = nn.BatchNorm1d(16, momentum=batch_momentum)

        self.linear1 = nn.Linear(4, 16, bias=self.linear_bias)

        if projection_axis == ProjectionAxis.both:
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
        -------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        if self.use_batch_norm is True:
            x = self.bn1(x)
        x = self.activation_function(x)
        x = self.linear2(x)
        return x


class OverProjNetLinear(OverProjNet):
    """
    Linear OverProjNet class
    """

    def __init__(self, projection_axis: ProjectionAxis = ProjectionAxis.x, activation: ActivationType = ActivationType.RELU,
                 init_w_normal: bool = False, use_batch_norm: bool = False, batch_momentum: float = 0.1):
        """
        Initialize the Linear OverProjNet class.

        Parameters
        ----------
        projection_axis: ProjectionAxis
            The projection axis
        activation: ActivationType
            Activation tag
        init_w_normal: bool
            Enables the normal distribution based weight initialization
        use_batch_norm: bool
            Enables batch normalization
        batch_momentum: float
            Momentum value used in batch normalization
        """
        super(OverProjNetLinear, self).__init__(activation=activation, use_batch_norm=use_batch_norm)

        if projection_axis == ProjectionAxis.both:
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
        -------
        x: torch.Tensor
            Output of the model
        """

        x = self.linear1(x)
        return x
