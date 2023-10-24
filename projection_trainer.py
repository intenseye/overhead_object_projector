from typing import Any, List, Tuple, Optional, Dict
import os.path
import random
import math
import json
import pickle
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from loss_model_utils.models import OverProjNet, OverProjNetXL, OverProjNetL, OverProjNetM, \
    OverProjNetS, OverProjNetXS, OverProjNetLinear, ProjectionAxis
from loss_model_utils.utils import str2bool, read_settings, seed_worker
from loss_model_utils.loss import Criterion_mse_loss, Criterion_nth_power_loss
from temp_params.temp_params_m import param_sweep as param_sweep

NUM_WORKERS = 0  # Number of workers to load data
THRESHOLD_CONST_FOR_HIT_PIXEL = 0.005  # Hit distance threshold between the prediction and ground truth (in pixels). The distance
# determines either a detection is true positive (if less than pr equal to the given threshold) or false positive etc.
THRESHOLD_FOR_HIT_DISTANCE = 0.5  # Hit distance threshold between the prediction and ground truth (in meters). The distance
# determines either a detection is true positive (if less than pr equal to the given threshold) or false positive etc.
FIXED_SEED_NUM = 6  # Seed number
ACTIVATE_CHECKPOINT_SAVE = False


def read_data(input_json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read input and target data produced

    Parameters
    ----------
    input_json_path: str
        Path to input file

    Returns
    -------
    dataset_pairs: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Read data from the txt file
    """
    with open(input_json_path, 'r') as f:
        data = f.read()
        file_content = json.loads(data)
        crane_positions = []
        projection_positions = []
        image_dims_unique = None
        for key in file_content:
            current_frame_info = file_content[key]
            if current_frame_info["any_labelled"] is True:
                for object_info in current_frame_info["objects"]:
                    if object_info["projection"]["x"] != -1:
                        crane_position = [object_info["bbox_coords"]["x_l"], object_info["bbox_coords"]["y_t"],
                                          object_info["bbox_coords"]["x_r"], object_info["bbox_coords"]["y_b"]]
                        crane_positions.append(crane_position)
                        projection_positions.append([object_info["projection"]["x"], object_info["projection"]["y"]])
                        if image_dims_unique is None:
                            image_dims_unique = [current_frame_info["image_dimensions"]["width"],
                                                 current_frame_info["image_dimensions"]["height"]]
                        else:
                            image_dims = [current_frame_info["image_dimensions"]["width"],
                                          current_frame_info["image_dimensions"]["height"]]
                            if image_dims[0] != image_dims_unique[0] or image_dims[1] != image_dims_unique[1]:
                                print('All image dimensions must be the same!')
                                sys.exit()

        data_read = np.array(crane_positions).astype(np.float32), np.array(projection_positions).astype(
            np.float32), np.array(image_dims).astype(np.float32)
        return data_read


def transform_data(inputs: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform input data and targets according to specific rules.

    Parameters
    ----------
    inputs: np.ndarray
        The input data, a 2D numpy array with shape (n_samples, n_features). It should have at least 4 columns.
    targets: np.ndarray
        The target data, a 2D numpy array with shape (n_samples, n_targets).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two 2D numpy arrays:
        - inputs_transformed: The transformed input data with the same shape as inputs.
        - targets_transformed: The transformed target data with the same shape as targets.
    """
    inputs_transformed = np.copy(inputs)
    inputs_transformed[:, 0] = (inputs[:, 0] + inputs[:, 2]) / 2
    inputs_transformed[:, 1] = inputs[:, 3]
    inputs_transformed[:, 2] = inputs[:, 2] - inputs[:, 0]
    inputs_transformed[:, 3] = inputs[:, 3] - inputs[:, 1]

    targets_transformed = targets - inputs_transformed[:, :2]
    return inputs_transformed, targets_transformed


def normalize_points(inputs: np.ndarray, targets: np.ndarray, image_size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize inputs and targets w.r.t the related dimension of the image

    Parameters
    ----------
    inputs: np.ndarray
        inputs
    targets: np.ndarray
        targets
    image_size: np.ndarray
        image_size

    Returns
    -------
    normalized_data: Tuple[np.ndarray, np.ndarray]
        Normalized data
    """
    inputs_norm = np.array(inputs / np.append(image_size, image_size)).astype(np.float32)
    targets_norm = np.array(targets / image_size).astype(np.float32)
    normalized_data = inputs_norm, targets_norm
    return normalized_data


class ProjectionTrainer:
    """
    Projection trainer class
    """

    def __init__(self, driver: str, config: ConfigParser):
        """
        Initialize the Projection trainer class.

        Parameters
        ----------
        driver: str
            Indicates driver to run the main code
        config: ConfigParser
            Configuration object containing the settings.
        """
        self.initialize_config(config)
        print('Overhead object projection training is started.')

        self.driver = driver
        self.network_size = param_sweep['network_size']  # Defines the size of the networks.
        self.test_model_mode = param_sweep['test_model_mode']  # Defines the size of the networks.

        if self.network_size == 'xl':
            overprojnet = OverProjNetXL
        elif self.network_size == 'l':
            overprojnet = OverProjNetL
        elif self.network_size == 'm':
            overprojnet = OverProjNetM
        elif self.network_size == 's':
            overprojnet = OverProjNetS
        elif self.network_size == 'xs':
            overprojnet = OverProjNetXS
        elif self.network_size == 'linear':
            overprojnet = OverProjNetLinear
        else:
            raise ValueError("Invalid network size %s" % repr(self.network_size))

        self.batch_size = int(float(param_sweep['batch_size']))  # Defines the batch size
        self.activation = param_sweep[
            'activation']  # Defines the activation function to be used in hidden layers of networks
        self.loss_function = param_sweep['loss_function_reg']  # Defines the loss function
        time_stamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        print('Model and log time stamp folder: ' + str(time_stamp))
        self.device = torch.device(self.device)
        self.logging_tool = self.logging_tool
        self.num_workers = NUM_WORKERS
        self.model_output_folder_path = os.path.join(self.main_output_folder, 'models', self.driver,
                                                     time_stamp)  # Model output folder
        self.log_folder_path = os.path.join(self.main_output_folder, 'logs', self.driver, time_stamp)  # Log folder

        self.fixed_partition_seed = str2bool(param_sweep['fixed_partition_seed'])  # Enables fixed seed mode
        self.use_mixed_precision = str2bool(param_sweep['use_mixed_precision'])  # Enables mixed precision operations.
        self.max_epoch = int(float(param_sweep['max_epoch']))  # Maximum number of training epoch.
        self.init_learning_rate = float(param_sweep[
                                            'init_learning_rate'])  # Initial learning rate (except for OneCycleLR. OneCycleLR uses this value as the maximum learning rate).
        self.scheduler_type = param_sweep[
            'scheduler_type']  # The scheduler type. ('reduce_plateau', 'lambda', 'one_cycle', 'step')
        self.betas = (float(param_sweep['betas_0']),  # Beta values used in AdamW optimizer.
                      float(param_sweep['betas_1']))
        self.weight_decay = float(param_sweep['weight_decay'])  # weight decay value used in AdamW optimizer.
        self.eps_adam = float(param_sweep['eps_adam'])  # epsilon value used in AdamW optimizer.

        if str(self.device) != 'cpu':
            torch.cuda.set_device(self.device)
        # fixed seed operation for reproducibility
        if self.fixed_partition_seed is True:
            os.environ['PYTHONHASHSEED'] = str(FIXED_SEED_NUM)
            random.seed(FIXED_SEED_NUM)
            np.random.seed(FIXED_SEED_NUM)
            torch.manual_seed(FIXED_SEED_NUM)
            torch.cuda.manual_seed(FIXED_SEED_NUM)
            torch.cuda.manual_seed_all(FIXED_SEED_NUM)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # folder creation if needed
        os.makedirs(self.model_output_folder_path, exist_ok=True)
        with open(os.path.join(self.model_output_folder_path, 'params.json'), 'w') as convert_file:
            convert_file.write('param_sweep = ')
            convert_file.write(json.dumps(param_sweep))
        os.makedirs(self.log_folder_path, exist_ok=True)
        with open(os.path.join(self.log_folder_path, 'params.json'), 'w') as convert_file:
            convert_file.write('param_sweep = ')
            convert_file.write(json.dumps(param_sweep))

        self.val_accuracy = 0.0
        self.validation_loss = math.inf
        self.best_accuracy = 0.0
        self.least_loss = math.inf
        self.init_epoch_count = 1
        self.iter_count = 0
        self.training_loss_at_min_val_loss = math.inf

        self.is_dist_map_enabled = False
        if os.path.isfile(self.auxiliary_data_path):
            self.load_distance_map(self.auxiliary_data_path)
            self.is_dist_map_enabled = True

        self.generator = torch.Generator()
        if self.fixed_partition_seed is True:
            self.generator.manual_seed(FIXED_SEED_NUM)
        self.train_ds, self.train_loader = self.initialize_dataloader(self.coordinates_file_train)
        self.val_ds, self.val_loader = self.initialize_dataloader(self.coordinates_file_val)
        if self.val_loader is not None:
            self.validation_enabled = True
        self.test_ds, self.test_loader = self.initialize_dataloader(self.coordinates_file_test)
        if self.test_loader is not None:
            self.test_enabled = True
        model = overprojnet(init_w_normal=self.init_w_normal, projection_axis=self.projection_axis,
                            use_batch_norm=self.use_batch_norm, batch_momentum=self.batch_momentum,
                            activation=self.activation)

        self.loss_plot_period = len(self.train_loader) // self.val_count_in_epoch
        self.model_save_period = 0
        if ACTIVATE_CHECKPOINT_SAVE is True:
            self.model_save_period = len(self.train_loader)

        optimizer = AdamW(model.parameters(), lr=self.init_learning_rate, betas=self.betas, eps=self.eps_adam,
                          weight_decay=self.weight_decay)
        if self.logging_tool == 'tensorboard':
            writer = SummaryWriter(log_dir=self.log_folder_path)
            self.init_model_optimizer_logger(model, optimizer, writer=writer)

        elif self.logging_tool == 'wandb':
            self.init_model_optimizer_logger(model, optimizer)
            self.initialize_wandb(time_stamp)

        if self.projection_axis == ProjectionAxis.x:
            self.related_cam_dim = self.image_size[0]
        elif self.projection_axis == ProjectionAxis.y:
            self.related_cam_dim = self.image_size[1]
        else:
            self.related_cam_dim = math.sqrt(self.image_size[0] ** 2 + self.image_size[1] ** 2)
        self.normalized_hit_thr = THRESHOLD_CONST_FOR_HIT_PIXEL

        self.denorm_coeff = torch.tensor([self.image_size[1], self.image_size[0]], dtype=torch.float32,
                                         device=self.device)
        self.best_model_path = None

    def initialize_config(self, config: ConfigParser):
        """
        Initialize and set the config fields.

        Parameters
        ----------
        config: ConfigParser
            Configuration object
        """
        self.device = config.get("projection_trainer", "DEVICE")
        self.apply_coord_transform = str2bool(config.get("projection_trainer", "APPLY_COORD_TRANSFORM"))
        self.logging_tool = config.get("projection_trainer", "LOGGING_TOOL")
        self.main_output_folder = config.get("projection_trainer", "MAIN_OUTPUT_FOLDER")
        self.init_w_normal = str2bool(config.get("projection_trainer", "INIT_W_NORMAL"))
        self.use_batch_norm = str2bool(config.get("projection_trainer", "USE_BATCH_NORM"))
        self.batch_momentum = float(config.get("projection_trainer", "BATCH_MOMENTUM"))
        self.val_count_in_epoch = int(float(config.get("projection_trainer", "VAL_COUNT_IN_EPOCH")))
        self.loss_patience = int(float(config.get("projection_trainer", "LOSS_PATIENCE")))
        self.loss_decrease_count = int(float(config.get("projection_trainer", "LOSS_DECREASE_COUNT")))
        self.loss_decrease_gamma = float(config.get("projection_trainer", "LOSS_DECREASE_GAMMA"))
        input_folder_path = config.get("projection_trainer", "INPUT_FOLDER_PATH")
        self.coordinates_file_train = os.path.join(input_folder_path, 'split', 'coordinates_train.json')
        self.coordinates_file_test = os.path.join(input_folder_path, 'split', 'coordinates_test.json')
        self.coordinates_file_val = os.path.join(input_folder_path, 'split', 'coordinates_val.json')
        self.auxiliary_data_path = os.path.join(input_folder_path, 'auxiliary_data.pickle')
        self.projection_axis = ProjectionAxis(config.get("projection_trainer", "PROJECTION_AXIS"))

    def load_distance_map(self, distance_map_path: str):
        """
        Load distance map and top-left coordinates from the pickle file loaded.

        Parameters
        ----------
        distance_map_path: str
            Path to distance map file
        """
        with open(distance_map_path, 'rb') as file:
            self.pixel_world_coords, self.dist_map_top_left_coord = pickle.load(file)
            # Since the elevation is fixed in entire area, only the z and x dimension of the distance map is used.
            self.pixel_world_coords = torch.from_numpy(self.pixel_world_coords[:, :, [0, 2]]).float().to(self.device)
            # Since we extended the original camera pixel dimensions, we need to use the new top-left location of the new camera.
            self.dist_map_top_left_coord = torch.from_numpy(self.dist_map_top_left_coord).to(self.device)

    def initialize_datasets(self, input_json_path: str):
        """
        Initialize the data loader after the normalization operation.

        Parameters
        ----------
        input_json_path: str
            Path to input file
        """

        inputs, targets, self.image_size = read_data(input_json_path)
        if self.apply_coord_transform is True:
            inputs, targets = transform_data(inputs, targets)
        inputs_norm, targets_norm = normalize_points(inputs, targets, self.image_size)
        if self.projection_axis == ProjectionAxis.x:
            targets_norm = np.expand_dims(targets_norm[:, 0], axis=1)
        elif self.projection_axis == ProjectionAxis.x:
            targets_norm = np.expand_dims(targets_norm[:, 1], axis=1)
        else:
            # Since the normalized distance (normalized coordinates) does not indicate the same amount of distance for
            # different dimensions of the cameras whose aspect ratio other than 1, we need to re-normalize them by
            # considering the pixel height and widths.
            targets_norm = targets_norm * np.array(self.image_size) / math.sqrt(
                self.image_size[0] ** 2 + self.image_size[1] ** 2)
        inputs_norm = torch.from_numpy(inputs_norm).float()
        targets_norm = torch.from_numpy(targets_norm).float()
        return TensorDataset(inputs_norm, targets_norm)

    def initialize_dataloader(self, data_json_path: str):
        """
        Initialize the data loader for the given data JSON file.

        Parameters
        ----------
        data_json_path (str): Path to the data JSON file.

        Returns
        -------
        Tuple[TensorDataset, Optional[DataLoader]]:
            A tuple containing the initialized dataset and data loader (if data is available).
        """
        data_ds = self.initialize_datasets(data_json_path)
        data_loader = None
        if len(data_ds) > 0:
            data_loader = DataLoader(
                data_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=self.generator,
            )
        return data_ds, data_loader

    def initialize_wandb(self, folder_name: str):
        """
        Initialize wandb (Weights & Biases) logging

        Parameters
        ----------
        folder_name: str
            sub folder name (timestamp) used in wandb run naming.
        """

        config_dict = param_sweep
        self.wandb_run = wandb.init(
            project='overhead_object_projection',
            name=folder_name,
            group=str(self.apply_coord_transform),
            job_type=self.driver,
            dir=self.log_folder_path,
            reinit=True,
        )

        wandb.config.update(config_dict)
        # metrics which are saved in each iteration
        wandb.define_metric('training_loss/iteration', summary='none')
        wandb.define_metric('validation_loss/iteration', summary='min')
        wandb.define_metric('validation_accuracy/iteration', summary='max')
        # best_validation_accuracy or least_validation_loss used as the optimization objective in Bayesian search.
        # best_validation_accuracy is non-decreasing and least_validation_loss is non-increasing metrics over time.
        wandb.define_metric('best_validation_accuracy/iteration')
        wandb.define_metric('least_validation_loss/iteration')
        wandb.define_metric('test_accuracy')
        wandb.define_metric('mean_test_pixel_error')
        wandb.define_metric('max_test_pixel_error')
        wandb.define_metric('mean_test_distance_error')
        wandb.define_metric('max_test_distance_error')
        for param_count in range(len(self.optimizer.param_groups)):
            wandb.define_metric('learning_rate_' + str(param_count) + '/iteration', summary='none')

    def init_model_optimizer_logger(self, model: OverProjNet, optimizer: AdamW,
                                    writer: Optional[SummaryWriter] = None):
        """
        Initialize the model, the optimizer and the logger (wandb or tensorboard)

        Parameters
        ----------
        model: FgSegNet_v2
            FgSegNet_v2 model
        optimizer: AdamW
            Learning rate optimizer
        writer: Optional[SummaryWriter]
            Logger writer either for wandb or tensorflow
        """

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.writer = writer

        if self.loss_function == 'mse':
            self.criterion = Criterion_mse_loss()
        if self.loss_function == 'min_max_error':
            self.criterion = Criterion_nth_power_loss(power_term=4)

        total_steps = self.max_epoch * len(self.train_loader)
        if self.scheduler_type == 'reduce_plateau':
            self.model_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.loss_patience,
                                                                     verbose=True)
        elif self.scheduler_type == 'lambda':  # time-based decay lambda function is used.
            lambda_lr = lambda iter_count: (float)(max(total_steps - iter_count, 0)) / (float)(total_steps)
            self.model_lr_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif self.scheduler_type == 'one_cycle':
            self.model_lr_scheduler = lr_scheduler.OneCycleLR(
                self.optimizer, total_steps=total_steps, max_lr=self.init_learning_rate
            )
        elif self.scheduler_type == 'step':
            lr_decrease_count = self.loss_decrease_count
            lr_dec_period = total_steps // (lr_decrease_count + 1)
            self.model_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=lr_dec_period,
                                                          gamma=self.loss_decrease_gamma)

        if self.use_mixed_precision is True:
            self.scaler = GradScaler()
        torch.cuda.empty_cache()
        print('Model is initialized.')

    def update_min_loss_summary(self):
        """
        Logs the training loss at the minimum validation loss point
        """
        if self.logging_tool == 'wandb':
            wandb.run.summary['training_loss_at_min_val_loss'] = self.training_loss_at_min_val_loss

    def save_log_iteration_training(self):
        """
        Logs the iteration based metrics during the training operations
        """
        if self.logging_tool == 'tensorboard':
            writer = self.writer
            writer.add_scalar('training_loss/iteration', self.iter_average_loss, self.iter_count)
            writer.add_scalar('validation_loss/iteration', self.validation_loss, self.iter_count)
            writer.add_scalar('validation_accuracy/iteration', self.val_accuracy, self.iter_count)
            writer.add_scalar('best_validation_accuracy/iteration', self.best_accuracy, self.iter_count)
            writer.add_scalar('least_validation_loss/iteration', self.least_loss, self.iter_count)
            for param_count, lr in enumerate(self.curr_learning_rate):
                writer.add_scalar('learning_rate_' + str(param_count) + '/iteration', lr, self.iter_count)
        elif self.logging_tool == 'wandb':
            wandb.log(
                {
                    'training_loss/iteration': self.iter_average_loss,
                    'validation_loss/iteration': self.validation_loss,
                    'validation_accuracy/iteration': self.val_accuracy,
                    'best_validation_accuracy/iteration': self.best_accuracy,
                    'least_validation_loss/iteration': self.least_loss,
                },
                step=self.iter_count,
            )
            for param_count, lr in enumerate(self.curr_learning_rate):
                wandb.log({'learning_rate_' + str(param_count) + '/iteration': lr}, step=self.iter_count)

    def save_log_iteration_test(self):
        """
        Logs the iteration based metrics during the testing operations
        """
        if self.logging_tool == 'tensorboard':
            writer = self.writer
            writer.add_scalar('mean_test_pixel_error', self.mean_test_pixel_error)
            writer.add_scalar('max_test_pixel_error', self.max_test_pixel_error)
            writer.add_scalar('test_accuracy_pixel', self.test_accuracy_pixel)
            if self.is_dist_map_enabled:
                writer.add_scalar('mean_test_distance_error', self.mean_test_distance_error)
                writer.add_scalar('max_test_distance_error', self.max_test_distance_error)
                writer.add_scalar('test_accuracy_distance', self.test_accuracy_distance)

        elif self.logging_tool == 'wandb':
            if self.is_dist_map_enabled:
                wandb.log(
                    {
                        'mean_test_pixel_error': self.mean_test_pixel_error,
                        'max_test_pixel_error': self.max_test_pixel_error,
                        'test_accuracy_pixel': self.test_accuracy_pixel,
                        'mean_test_distance_error': self.mean_test_distance_error,
                        'max_test_distance_error': self.max_test_distance_error,
                        'test_accuracy_distance': self.test_accuracy_distance,
                    },
                    step=self.iter_count,
                )
            else:
                wandb.log(
                    {
                        'mean_test_pixel_error': self.mean_test_pixel_error,
                        'max_test_pixel_error': self.max_test_pixel_error,
                        'test_accuracy_pixel': self.test_accuracy_pixel,
                    },
                    step=self.iter_count,
                )

    def save_checkpoint(
            self, model_state_dict: Dict[str, torch.Tensor], optimizer_state_dict: Dict[Any, Any], epoch: int,
            iteration: int, accuracy: float, loss: float, mode: Optional[str] = None
    ):
        """
        Save the checkpoint model with additional metadata

        Parameters
        ----------
        model_state_dict: Dict[str, torch.Tensor]
            State dictionary of the model
        optimizer_state_dict: Dict[Any, Any]
           State dictionary of the optimizer
        epoch: int
            Epoch count that the saved model is obtained
        iteration: int
            Iteration count that the saved model is obtained
        accuracy: float
            Measured accuracy of the model
        loss: float
            Measured loss of the model
        mode: Optional[str]
            Mode denoting the state of the save criteria

        """
        if mode == 'best_accuracy':
            filename = 'best_accuracy_model.pth'
            self.best_model_path = os.path.join(self.model_output_folder_path, filename)
        elif mode == 'least_loss':
            filename = 'least_loss_model.pth'
            self.best_model_path = os.path.join(self.model_output_folder_path, filename)
        else:
            filename = '-{0:08d}.pth'.format(self.iter_count)

        checkpoint_model_path = os.path.join(self.model_output_folder_path, filename)
        torch.save(
            {
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'epoch': epoch,
                'iteration': iteration,
                'accuracy': accuracy,
                'loss': loss,
            },
            checkpoint_model_path,
        )
        if mode == 'best_accuracy' or mode == 'least_loss':
            print(f'\nModel and optimizer parameters for the best performed iteration are saved in the {filename}.\n')
        else:
            print(
                f'Model and optimizer parameters for the {self.iter_count}\'th iteration are saved in the {filename}.')

    def validation(self):
        """
        Applies a validation step
        """
        with torch.no_grad():
            sample_count = 0
            cum_validation_loss = 0
            cum_hits = 0

            for sample in self.val_loader:
                obj_coords, projection_coord = sample
                obj_coords = obj_coords.type(torch.FloatTensor).to(self.device, non_blocking=True)
                projection_coord = projection_coord.type(torch.FloatTensor).to(self.device, non_blocking=True)

                if self.use_mixed_precision is True:
                    with autocast():
                        output = self.model(obj_coords)
                        loss = self.criterion(output, projection_coord)
                else:
                    output = self.model(obj_coords)
                    loss = self.criterion(output, projection_coord)
                abs_diffs = torch.linalg.norm(torch.abs(output - projection_coord), dim=1)
                hit_count = torch.sum((abs_diffs <= self.normalized_hit_thr).to(int))

                sample_count += abs_diffs.size(dim=0)
                cum_hits += hit_count.item()
                cum_validation_loss += torch.sum(loss).item()

            self.validation_loss = cum_validation_loss / sample_count
            print('Validation (iter {:8d}, loss {:.6f})'.format(self.iter_count, self.validation_loss))

            self.val_accuracy = cum_hits / len(self.val_ds)
            print(
                'Max validation (accuracy) in iteration {:d} : ({:.6f})'.format(
                    self.iter_count, self.val_accuracy
                )
            )

    def train_forward(self, object_coords, projection_coord) -> Tuple[float, int]:
        """
        Applies forward pass, loss calculations and optimizer updates in training phase

        Parameters
        ----------
        object_coords: torch.Tensor
            Coordinates of the object
        projection_coord: torch.Tensor
            Coordinates of the projection point

        Returns
        -------
        forward_output: Tuple[float, int]
            Calculated loss value and sample count
        """
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_mixed_precision is True:
            with autocast():
                output = self.model(object_coords)
                loss = self.criterion(output, projection_coord)

            self.scaler.scale(loss.mean()).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(object_coords)
            loss = self.criterion(output, projection_coord)
            loss.mean().backward()
            self.optimizer.step()
        self.iter_count += 1
        cum_loss = torch.sum(loss).item()
        sample_count = loss.size(dim=0)
        return cum_loss, sample_count

    def step(self, sample: List[torch.tensor]) -> Tuple[float, int]:
        """
        Conducts simple conversions and then one-step training pass.

        Parameters
        ----------
        sample: List[torch.tensor]
            A sample loaded from dataloader

        Returns
        -------
        forward_output: -> Tuple[float, int]
            Calculated loss value and sample count
        """
        obj_coords, projection_coord = sample
        obj_coords = obj_coords.type(torch.FloatTensor).to(self.device, non_blocking=True)
        projection_coord = projection_coord.type(torch.FloatTensor).to(self.device, non_blocking=True)
        return self.train_forward(obj_coords, projection_coord)

    def run_validation(self):
        """
        Main validation function.
        """
        self.model.eval()
        self.validation()

        if self.validation_loss < self.least_loss:
            self.least_loss = self.validation_loss
            self.training_loss_at_min_val_loss = self.iter_average_loss
            self.update_min_loss_summary()

            self.save_checkpoint(
                self.model.state_dict(),
                self.optimizer.state_dict(),
                self.epoch_count,
                self.iter_count,
                self.val_accuracy,
                self.least_loss,
                mode='least_loss',
            )

        if self.val_accuracy > self.best_accuracy:
            self.best_accuracy = self.val_accuracy
            self.save_checkpoint(
                self.model.state_dict(),
                self.optimizer.state_dict(),
                self.epoch_count,
                self.iter_count,
                self.best_accuracy,
                self.validation_loss,
                mode='best_accuracy',
            )

        self.save_log_iteration_training()

    def run_train(self):
        """
        Main train function.
        """
        for self.epoch_count in range(self.init_epoch_count, self.max_epoch + 1):
            epoch_loss = 0
            iter_loss = 0
            iter_sample_count = 0
            epoch_sample_count = 0

            self.curr_learning_rate = [None] * len(self.optimizer.param_groups)

            print('\nEpoch {}/{}'.format(self.epoch_count, self.max_epoch))
            print('-' * 50)

            for sample in tqdm(self.train_loader):
                self.model.train()
                batch_loss, batch_sample_count = self.step(sample)
                if (
                        self.scheduler_type == 'one_cycle'
                        or self.scheduler_type == 'lambda'
                        or self.scheduler_type == 'step'
                ):
                    self.model_lr_scheduler.step()
                iter_loss += batch_loss
                epoch_loss += batch_loss
                iter_sample_count += batch_sample_count
                epoch_sample_count += batch_sample_count

                if self.loss_plot_period > 0 and self.iter_count % self.loss_plot_period == 0:
                    self.iter_average_loss = iter_loss / iter_sample_count

                    print('\nTraining   (iter {:8d}, loss {:.6f})'.format(self.iter_count, self.iter_average_loss))
                    for param_count, param_group in enumerate(self.optimizer.param_groups):
                        self.curr_learning_rate[param_count] = param_group['lr']
                    iter_loss = 0
                    iter_sample_count = 0
                    if self.validation_enabled:
                        self.run_validation()
                        if self.scheduler_type == 'reduce_plateau':
                            self.model_lr_scheduler.step(self.validation_loss)

                if self.model_save_period > 0 and self.iter_count % self.model_save_period == 0:
                    self.save_checkpoint(
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        self.epoch_count,
                        self.iter_count,
                        self.val_accuracy,
                        self.validation_loss,
                    )

            self.epoch_average_loss = epoch_loss / epoch_sample_count

            print('\n' + '-' * 25)
            print('Training      (EPOCH {:4d}, loss {:.6f})'.format(self.epoch_count, self.epoch_average_loss))
            for param_count, param_group in enumerate(self.optimizer.param_groups):
                self.curr_learning_rate[param_count] = param_group['lr']
                print(
                    'Learning rate (EPOCH {:4d}, lr   {:.8f})'.format(
                        self.epoch_count, self.curr_learning_rate[param_count]
                    )
                )
        if self.validation_enabled is True:
            self.run_validation()
        if self.model_save_period > 0:
            self.save_checkpoint(
                self.model.state_dict(),
                self.optimizer.state_dict(),
                self.epoch_count,
                self.iter_count,
                self.val_accuracy,
                self.validation_loss,
            )

        print('-' * 50)
        print('Best validation accuracy: {:6f}'.format(self.best_accuracy))
        if self.logging_tool == 'tensorboard':
            self.writer.close()

    def read_model_file(self):
        """
        Reads the printed best model.
        """
        print('-' * 50)
        print("Reading the projection estimation model ")
        self.checkpoint = torch.load(os.path.join(self.model_output_folder_path, self.test_model_mode + '.pth'), map_location=torch.device(self.device))

    def run_test(self):
        """
        Main test function.
        """
        if self.test_enabled and self.best_model_path is not None:
            self.read_model_file()
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)

            with torch.no_grad():
                self.model.eval()
                cum_test_loss = 0
                test_sample_count = 0
                cum_abs_diff = 0
                cum_test_hit_count_pixel = 0
                max_diff = 0

                cum_abs_diff_distance = 0
                cum_test_hit_count_distance = 0
                self.max_test_distance_error = 0

                for sample in self.test_loader:
                    obj_coords, projection_coord = sample
                    obj_coords = obj_coords.type(torch.FloatTensor).to(self.device, non_blocking=True)
                    projection_coord = projection_coord.type(torch.FloatTensor).to(self.device, non_blocking=True)

                    if self.use_mixed_precision is True:
                        with autocast():
                            output = self.model(obj_coords)
                            loss = self.criterion(output, projection_coord)
                    else:
                        output = self.model(obj_coords)
                        loss = self.criterion(output, projection_coord)

                    abs_diffs_pixel = torch.linalg.norm(torch.abs(output - projection_coord), dim=1)
                    hit_count_pixel = torch.sum((abs_diffs_pixel <= self.normalized_hit_thr).to(int))
                    cum_test_hit_count_pixel += hit_count_pixel.item()

                    cum_test_loss += torch.sum(loss).item()
                    test_sample_count += loss.size(dim=0)

                    cum_abs_diff += torch.sum(abs_diffs_pixel).item()
                    if torch.max(abs_diffs_pixel).item() > max_diff:
                        max_diff = torch.max(abs_diffs_pixel).item()

                    if self.is_dist_map_enabled is True:
                        denorm_output = torch.round((output + obj_coords[:, :2]) * self.denorm_coeff).int()
                        denorm_pred = torch.round((projection_coord + obj_coords[:, :2]) * self.denorm_coeff).int()
                        denorm_output_shifted = denorm_output - self.dist_map_top_left_coord
                        denorm_pred_shifted = denorm_pred - self.dist_map_top_left_coord

                        dist = torch.zeros([denorm_output_shifted.shape[0], 2], dtype=torch.float32, device=self.device)
                        for i in range(denorm_output_shifted.shape[0]):
                            dist[i] = self.pixel_world_coords[denorm_output_shifted[i][1], denorm_output_shifted[i][0],
                                      :] - \
                                      self.pixel_world_coords[denorm_pred_shifted[i][1], denorm_pred_shifted[i][0], :]
                        abs_diffs_distance = torch.linalg.norm(dist, dim=1)
                        hit_count_distance = torch.sum((abs_diffs_distance <= THRESHOLD_FOR_HIT_DISTANCE).to(int))
                        cum_test_hit_count_distance += hit_count_distance.item()

                        cum_abs_diff_distance += torch.sum(abs_diffs_distance).item()
                        if torch.max(abs_diffs_distance).item() > self.max_test_distance_error:
                            self.max_test_distance_error = torch.max(abs_diffs_distance).item()

                self.test_loss = cum_test_loss / test_sample_count
                print('Test loss {:.6f}'.format(self.test_loss))
                self.test_accuracy_pixel = cum_test_hit_count_pixel / len(self.test_ds)
                print('Test accuracy (pixel): ({:.6f})'.format(self.test_accuracy_pixel))
                self.mean_test_pixel_error = (cum_abs_diff * self.related_cam_dim) / len(self.test_ds)
                print('Mean test error (pixel): ({:.6f})'.format(self.mean_test_pixel_error))
                self.max_test_pixel_error = max_diff * self.related_cam_dim
                print('Max test error (pixel): ({:.6f})'.format(self.max_test_pixel_error))

                if self.is_dist_map_enabled is True:
                    self.test_accuracy_distance = cum_test_hit_count_distance / len(self.test_ds)
                    print('Test accuracy (distance): ({:.6f})'.format(self.test_accuracy_distance))
                    self.mean_test_distance_error = cum_abs_diff_distance / len(self.test_ds)
                    print('Mean test error (distance): ({:.6f})'.format(self.mean_test_distance_error))
                    print('Max test error (distance): ({:.6f})'.format(self.max_test_distance_error))

                self.save_log_iteration_test()
                if self.logging_tool == 'tensorboard':
                    self.writer.close()
                elif self.logging_tool == 'wandb':
                    self.wandb_run.finish()
        else:
            print('No best model to perform testing!')


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to train the projection_model")
    parser.add_argument("--driver", help="Indicates driver to run the main code", choices=['wandb', 'manual'],
                        default='manual')
    parser.add_argument("--settings_path", help="Path to the settings file.", default=r"./settings/settings_1.ini")

    args = parser.parse_args()
    driver_ = args.driver
    config_ = read_settings(args.settings_path)

    # for sets in [1, 2, 3, 4, 5]:
    #     # for dev_mode in ['bbox_dev', 'both_dev', 'no_dev', 'proj_dev']:
    #     for dev_mode in ['both_dev']:
    #         input_dir_path = "/home/poyraz/intenseye/input_outputs/overhead_object_projector/datasets/OverheadSimIntenseye/Set0" + str(sets) + "/" + dev_mode
    #         config_.set("projection_trainer", "INPUT_FOLDER_PATH", input_dir_path)
    #         # for apply_trains in ["True", "False"]:
    #         for apply_trains in ["True"]:
    #             config_.set("projection_trainer", "APPLY_COORD_TRANSFORM", apply_trains)
    #             out_dir_path = "/home/poyraz/intenseye/input_outputs/overhead_object_projector/models/OverheadSimIntenseye_mod/Set0" + str(sets) + "/" + dev_mode + "/" + apply_trains + "/m_pow4"
    #             config_.set("projection_trainer", "MAIN_OUTPUT_FOLDER", out_dir_path)
    #             proj_trainer = ProjectionTrainer(driver=driver_, config=config_)
    #             proj_trainer.run_train()
    #             proj_trainer.run_test()

    # for sets in [1, 2]:
    #     input_dir_path = "/home/poyraz/intenseye/input_outputs/overhead_object_projector/datasets/CraneIntenseye/Set0" + str(sets)
    #     config_.set("projection_trainer", "INPUT_FOLDER_PATH", input_dir_path)
    #     for apply_trains in ["True", "False"]:
    #         config_.set("projection_trainer", "APPLY_COORD_TRANSFORM", apply_trains)
    #         out_dir_path = "/home/poyraz/intenseye/input_outputs/overhead_object_projector/models/OverheadIntenseye_mod/Set0" + str(sets) + "/" + apply_trains + "/m_pow4"
    #         config_.set("projection_trainer", "MAIN_OUTPUT_FOLDER", out_dir_path)
    #         proj_trainer = ProjectionTrainer(driver=driver_, config=config_)
    #         proj_trainer.run_train()
    #         proj_trainer.run_test()

    proj_trainer = ProjectionTrainer(driver=driver_, config=config_)
    proj_trainer.run_train()
    proj_trainer.run_test()
