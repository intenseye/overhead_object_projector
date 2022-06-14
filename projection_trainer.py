import os.path
import random
import math
import json
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from temp_params import *
from argparse import ArgumentParser
from models import RegressionModel


DEVICE = "cuda:0"  # 'cuda:0' or 'cpu'
LOGGING_TOOL = 'wandb'  # 'wandb' or 'tensorboard'
NUM_WORKERS = 0
PIXEL_THRESHOLD_FOR_HIT = 5
FIXED_SEED_NUM = 35
VAL_COUNT_IN_EPOCH = 4
LOSS_DECREASE_COUNT = 4
LOSS_DECREASE_GAMMA = 0.1
LOSS_PATIENCE = 2
MAIN_OUTPUT_FOLDER = '/home/poyraz/intenseye/input_outputs/crane_simulation/'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str2bool(bool_argument):
    if bool_argument.lower() == "true":
        return True
    elif bool_argument.lower() == "false":
        return False
    else:
        raise ValueError


def read_input_txt(input_txt_path):
    with open(input_txt_path, 'r') as f:
        input_list = []
        output_list = []
        image_size = None
        for rows_id, rows in enumerate(raw.strip().split() for raw in f):
            if rows_id > 0:
                input_list.append(rows[:4])
                output_list.append(rows[4:6])
            if rows_id == 1:
                image_size = np.array(rows[6:]).astype(np.float32)

    return np.array(input_list).astype(np.float32), np.array(output_list).astype(np.float32), image_size


def normalize_points(inputs, targets, image_size):
    inputs_norm = np.array(inputs / np.append(image_size, image_size)).astype(np.float32)
    targets_norm = np.array(targets / image_size).astype(np.float32)
    return inputs_norm, targets_norm


class Criterion_mse_loss(nn.Module):
    def __init__(self):
        super(Criterion_mse_loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        loss = self.loss(output, target)
        return loss


class ProjectionTrainer:
    def __init__(self, driver, input_txt_path, projection_axis):
        print('Overhead object projection training is started.')
        self.projection_axis = projection_axis
        self.driver = driver
        self.batch_size = int(param_sweep['batch_size'])
        self.loss_function = param_sweep['loss_function_reg']
        time_stamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        print('Model and log time stamp folder: ' + str(time_stamp))
        self.device = torch.device(DEVICE)
        self.logging_tool = LOGGING_TOOL
        self.num_workers = NUM_WORKERS
        self.model_output_folder_path = os.path.join(MAIN_OUTPUT_FOLDER, 'models', self.driver, time_stamp)
        self.log_folder_path = os.path.join(MAIN_OUTPUT_FOLDER, 'logs', self.driver, time_stamp)

        self.tuning_method = param_sweep['tuning_method']
        self.fixed_partition_seed = str2bool(param_sweep['fixed_partition_seed'])
        self.validation_ratio = float(param_sweep['validation_ratio'])
        self.test_ratio = float(param_sweep['test_ratio'])

        self.zero_mean_enabled = str2bool(param_sweep['zero_mean_enabled'])
        self.use_mixed_precision = str2bool(param_sweep['use_mixed_precision'])
        self.max_epoch = int(float(param_sweep['max_epoch']))
        self.init_learning_rate = float(param_sweep['init_learning_rate'])
        self.scheduler_type = param_sweep['scheduler_type']
        self.betas = (float(param_sweep['betas_0']),
                      float(param_sweep['betas_1']))
        self.weight_decay = float(param_sweep['weight_decay'])
        self.eps_adam = float(param_sweep['eps_adam'])
        self.init_w_normal = str2bool(param_sweep['init_w_normal'])

        if str(self.device) != 'cpu':
            torch.cuda.set_device(self.device)
        # seed
        if self.fixed_partition_seed:
            os.environ['PYTHONHASHSEED'] = str(FIXED_SEED_NUM)
            random.seed(FIXED_SEED_NUM)
            np.random.seed(FIXED_SEED_NUM)
            torch.manual_seed(FIXED_SEED_NUM)
            torch.cuda.manual_seed(FIXED_SEED_NUM)
            torch.cuda.manual_seed_all(FIXED_SEED_NUM)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if not os.path.exists(self.model_output_folder_path):
            os.makedirs(self.model_output_folder_path)
            with open(os.path.join(self.model_output_folder_path, 'params.json'), 'w') as convert_file:
                convert_file.write('param_sweep = ')
                convert_file.write(json.dumps(param_sweep))
        if not os.path.exists(self.log_folder_path):
            os.makedirs(self.log_folder_path)
            with open(os.path.join(self.log_folder_path, 'params.json'), 'w') as convert_file:
                convert_file.write('param_sweep = ')
                convert_file.write(json.dumps(param_sweep))

        self.val_accuracy = 0.0
        self.validation_loss = 1.0
        self.best_accuracy = 0.0
        self.least_loss = 1.0
        self.init_epoch_count = 1
        self.iter_count = 0
        self.min_val_loss = math.inf
        self.training_loss_at_min_val_loss = math.inf

        self.initialize_dataloaders(input_txt_path)
        model = RegressionModel(init_w_normal=self.init_w_normal)

        self.loss_plot_period = len(self.train_loader)//VAL_COUNT_IN_EPOCH
        self.model_save_period = 0
        if self.validation_ratio == 0:
            self.model_save_period = len(self.train_loader)

        optimizer = AdamW(model.parameters(), lr=self.init_learning_rate, betas=self.betas, eps=self.eps_adam, weight_decay=self.weight_decay)
        if self.logging_tool == 'tensorboard':
            writer = SummaryWriter(log_dir=self.log_folder_path)
            self.init_model_optimizer_logger(model, optimizer, writer=writer)

        elif self.logging_tool == 'wandb':
            self.init_model_optimizer_logger(model, optimizer)
            self.initialize_wandb(time_stamp)

        if projection_axis == 'x':
            related_cam_dim = self.image_size[0]
        else:
            related_cam_dim = self.image_size[1]
        self.normalized_hit_thr = PIXEL_THRESHOLD_FOR_HIT / related_cam_dim

        self.best_model_path = None

    def dataset_splitter(self, dataset):

        input_size = len(dataset)
        val_input_size = int(input_size * self.validation_ratio)
        test_input_size = int(input_size * self.test_ratio)
        train_input_size = input_size - (val_input_size + test_input_size)

        random_indices = random.sample(range(input_size), input_size)
        val_indices = random_indices[0:val_input_size]
        test_indices = random_indices[val_input_size:test_input_size + val_input_size]
        train_indices = random_indices[-train_input_size:]

        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)
        train_ds = Subset(dataset, train_indices)
        return train_ds, val_ds, test_ds

    def initialize_dataloaders(self, input_txt_path):
        inputs, targets, self.image_size = read_input_txt(input_txt_path)
        inputs_norm, targets_norm = normalize_points(inputs, targets, self.image_size)
        inputs_norm = torch.from_numpy(inputs_norm)
        if self.projection_axis == 'x':
            targets_norm = torch.from_numpy(targets_norm[:, 0])
        else:
            targets_norm = torch.from_numpy(targets_norm[:, 1])
        self.train_ds = TensorDataset(inputs_norm, targets_norm)
        self.generator = torch.Generator()
        if self.fixed_partition_seed:
            self.generator.manual_seed(FIXED_SEED_NUM)

        self.validation_enabled = False
        self.val_ds = None
        self.test_ds = None
        if (self.validation_ratio + self.test_ratio) > 0:
            self.train_ds, self.val_ds, self.test_ds = self.dataset_splitter(self.train_ds)
            if len(self.val_ds) > 0:
                self.validation_enabled = True
            if len(self.test_ds) > 0:
                self.test_enabled = True

        if self.validation_enabled:
            print('Validation operations are enabled with %' + str(self.validation_ratio * 100) + ' of data.')
        else:
            print('Validation operations are disabled.')

        if self.test_enabled:
            print('Test operations are enabled with %' + str(self.test_ratio * 100) + ' of data.')
        else:
            print('Test operations are disabled.')

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    def initialize_wandb(self, folder_name):
        config_dict = param_sweep
        wandb.init(
            project='overhead_object_projection',
            name=folder_name,
            group=self.projection_axis,
            job_type=self.driver,
            dir=self.log_folder_path,
        )

        wandb.config.update(config_dict)
        # metrics which are saved in each iteration
        wandb.define_metric('training_loss/iteration', summary='none')
        wandb.define_metric('validation_loss/iteration', summary='min')
        wandb.define_metric('validation_accuracy/iteration', summary='max')
        # best_validation_accuracy is used as the optimization objective in Bayesian search
        wandb.define_metric('best_validation_accuracy/iteration')
        for param_count in range(len(self.optimizer.param_groups)):
            wandb.define_metric('learning_rate_' + str(param_count) + '/iteration', summary='none')

    def init_model_optimizer_logger(self, model, optimizer, writer=None):
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.writer = writer

        if self.loss_function == 'mse':
            self.criterion = Criterion_mse_loss()

        total_steps = self.max_epoch * len(self.train_loader)
        if self.scheduler_type == 'reduce_plateau':
            self.model_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=LOSS_PATIENCE, verbose=True)
        elif self.scheduler_type == 'lambda':
            lambda_lr = lambda iter_count: (float)(max(total_steps - iter_count, 0)) / (float)(total_steps)
            self.model_lr_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif self.scheduler_type == 'one_cycle':
            self.model_lr_scheduler = lr_scheduler.OneCycleLR(
                self.optimizer, total_steps=total_steps, max_lr=self.init_learning_rate
            )
        elif self.scheduler_type == 'step':
            lr_decrease_count = LOSS_DECREASE_COUNT
            lr_dec_period = total_steps // (lr_decrease_count + 1)
            self.model_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=lr_dec_period, gamma=LOSS_DECREASE_GAMMA)

        if self.use_mixed_precision:
            self.scaler = GradScaler()
        torch.cuda.empty_cache()
        print('Model is initialized.')

    def update_min_loss_summary(self):
        if self.logging_tool == 'wandb':
            wandb.run.summary['training_loss_at_min_val_loss'] = self.training_loss_at_min_val_loss

    def save_log_iteration(self):
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

    def save_checkpoint(
            self, model_state_dict, optimizer_state_dict, epoch, iteration, accuracy, loss, is_best=False
    ):
        if is_best:
            filename = 'best_model.pth'
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
        if is_best:
            print(f'\nModel and optimizer parameters for the best performed iteration are saved in the {filename}.\n')
        else:
            print(f'Model and optimizer parameters for the {self.iter_count}\'th iteration are saved in the {filename}.')

    def validation(self):
        with torch.no_grad():
            cum_validation_loss = 0
            cum_accuracy = 0

            for sample in self.val_loader:
                obj_coords, projection_coord = sample
                obj_coords = obj_coords.type(torch.FloatTensor).to(self.device, non_blocking=True)
                projection_coord = projection_coord.type(torch.FloatTensor).to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    with autocast():
                        output = self.model(obj_coords)
                        loss = self.criterion(output[:, 0], projection_coord)
                else:
                    output = self.model(obj_coords)
                    loss = self.criterion(output[:, 0], projection_coord)
                abs_diffs = torch.abs(output[:, 0] - projection_coord)
                hit_count = torch.sum((abs_diffs <= self.normalized_hit_thr).to(int))
                accuracy = torch.div(hit_count, torch.numel(projection_coord))

                cum_accuracy += accuracy.item()
                cum_validation_loss += loss.item()

            self.validation_loss = cum_validation_loss / len(self.val_loader)
            print('Validation (iter {:8d}, loss {:.6f})'.format(self.iter_count, self.validation_loss))

            self.val_accuracy = cum_accuracy / len(self.val_loader)
            print(
                'Max validation (accuracy) in iteration {:d} : ({:.6f})'.format(
                    self.iter_count, self.val_accuracy
                )
            )

    def train_forward(self, object_coords, projection_coord):
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_mixed_precision:
            with autocast():
                output = self.model(object_coords)
                loss = self.criterion(output[:, 0], projection_coord)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(object_coords)
            loss = self.criterion(output[:, 0], projection_coord)
            loss.backward()
            self.optimizer.step()
        self.iter_count += 1
        return loss.item()

    def step(self, sample):
        obj_coords, projection_coord = sample
        obj_coords = obj_coords.type(torch.FloatTensor).to(self.device, non_blocking=True)
        projection_coord = projection_coord.type(torch.FloatTensor).to(self.device, non_blocking=True)
        return self.train_forward(obj_coords, projection_coord)

    def apply_validation(self):
        self.model.eval()
        self.validation()

        save_flag = False
        if self.validation_loss < self.least_loss:
            self.least_loss = self.validation_loss
            if self.tuning_method == 'loss':
                save_flag = True
        if self.val_accuracy > self.best_accuracy:
            self.best_accuracy = self.val_accuracy
            if self.tuning_method == 'accuracy':
                save_flag = True

        if save_flag:
            self.save_checkpoint(
                self.model.state_dict(),
                self.optimizer.state_dict(),
                self.epoch_count,
                self.iter_count,
                self.best_accuracy,
                self.least_loss,
                True,
            )

        if self.validation_loss < self.min_val_loss:
            self.min_val_loss = self.validation_loss
            self.training_loss_at_min_val_loss = self.iter_average_loss
            self.update_min_loss_summary()

        self.save_log_iteration()

    def train(self):
        for self.epoch_count in range(self.init_epoch_count, self.max_epoch + 1):
            self.epoch_loss = []
            iter_loss = []
            self.curr_learning_rate = [None] * len(self.optimizer.param_groups)

            print('\nEpoch {}/{}'.format(self.epoch_count, self.max_epoch))
            print('-' * 50)

            for sample in tqdm(self.train_loader):
                self.model.train()
                loss_item = self.step(sample)
                if (
                        self.scheduler_type == 'one_cycle'
                        or self.scheduler_type == 'lambda'
                        or self.scheduler_type == 'step'
                ):
                    self.model_lr_scheduler.step()
                iter_loss.append(loss_item)
                self.epoch_loss.append(loss_item)
                if self.loss_plot_period > 0 and self.iter_count % self.loss_plot_period == 0:
                    self.iter_average_loss = sum(iter_loss) / len(iter_loss)

                    print('\nTraining   (iter {:8d}, loss {:.6f})'.format(self.iter_count, self.iter_average_loss))
                    for param_count, param_group in enumerate(self.optimizer.param_groups):
                        self.curr_learning_rate[param_count] = param_group['lr']
                    iter_loss.clear()
                    if self.validation_enabled:
                        self.apply_validation()
                        if self.scheduler_type == 'reduce_plateau':
                            self.model_lr_scheduler.step(self.validation_loss)

                if (self.model_save_period > 0 and self.iter_count % self.model_save_period == 0):
                    self.save_checkpoint(
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        self.epoch_count,
                        self.iter_count,
                        self.val_accuracy,
                        self.validation_loss,
                    )

            self.epoch_average_loss = sum(self.epoch_loss) / len(self.epoch_loss)
            self.epoch_loss.clear()

            print('\n' + '-' * 25)
            print('Training      (EPOCH {:4d}, loss {:.6f})'.format(self.epoch_count, self.epoch_average_loss))
            for param_count, param_group in enumerate(self.optimizer.param_groups):
                self.curr_learning_rate[param_count] = param_group['lr']
                print(
                    'Learning rate (EPOCH {:4d}, lr   {:.8f})'.format(
                        self.epoch_count, self.curr_learning_rate[param_count]
                    )
                )
        if self.validation_enabled:
            self.apply_validation()
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

    def run_train(self):
        self.train()

    def read_model_file(self):
        print('-' * 50)
        print("Reading the projection estimation model ")
        self.checkpoint = torch.load(self.best_model_path, map_location=torch.device(self.device))

    def run_test(self):
        if self.best_model_path is not None:
            self.read_model_file()
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)

            with torch.no_grad():
                self.model.eval()
                cum_test_loss = 0
                cum_test_loss_accuracy = 0

                for sample in self.test_loader:
                    obj_coords, projection_coord = sample
                    obj_coords = obj_coords.type(torch.FloatTensor).to(self.device, non_blocking=True)
                    projection_coord = projection_coord.type(torch.FloatTensor).to(self.device, non_blocking=True)

                    if self.use_mixed_precision:
                        with autocast():
                            output = self.model(obj_coords)
                            loss = self.criterion(output[:, 0], projection_coord)
                    else:
                        output = self.model(obj_coords)
                        loss = self.criterion(output[:, 0], projection_coord)
                    abs_diffs = torch.abs(output[:, 0] - projection_coord)
                    hit_count = torch.sum((abs_diffs <= self.normalized_hit_thr).to(int))
                    accuracy = torch.div(hit_count, torch.numel(projection_coord))

                    cum_test_loss_accuracy += accuracy.item()
                    cum_test_loss += loss.item()

                self.test_loss = cum_test_loss / len(self.test_loader)
                print('Test (iter {:8d}, loss {:.6f})'.format(self.iter_count, self.test_loss))

                self.test_accuracy = cum_test_loss_accuracy / len(self.test_loader)
                print(
                    'Max test (accuracy) in iteration {:d} : ({:.6f})'.format(
                        self.iter_count, self.test_accuracy
                    )
                )
        else:
            print('No best model to perform testing!')


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to train the projection_model")
    parser.add_argument("--driver", help="Indicates driver", default='manual')
    parser.add_argument("--input_txt_path", help="Path to input txt file.", default='/home/poyraz/intenseye/input_outputs/crane_simulation/inputs_outputs_w_roll_dev_non_norm.txt')
    parser.add_argument("--projection_axis", help="Indicates axis of the projection", choices=['x', 'y'], default='x')

    args = parser.parse_args()
    driver = args.driver
    input_txt_path = args.input_txt_path
    projection_axis = args.projection_axis
    proj_trainer = ProjectionTrainer(driver=driver, input_txt_path=input_txt_path, projection_axis=projection_axis)
    proj_trainer.run_train()
    proj_trainer.run_test()
