# Overhead object projector: OverProjNet
### OverProjNet is simple, fast and reliable estimator for projection points of overhead objects. 

This repo is used to train and test projection points of overhead objects from images. In addition to that, simulation 
samples can be generated in the desired configuration.

> [**Overhead object projector: OverProjNet**](https://www.sciencedirect.com/science/article/pii/S2667305323000947)
> 
> [Poyraz Umut Hatipoglu](mailto:poyraz@intenseye.com?subject=OverProjNet), [Ali Ufuk Yaman](mailto:ufuk@intenseye.com?subject=OverProjNet), [Okan Ulusoy](mailto:okan@intenseye.com?subject=OverProjNet)
> 
> News: (08. 2023) Our paper is accepted by Intelligent Systems with Applications: https://doi.org/10.1016/j.iswa.2023.200269 

## Highlights
- Overhead object projection is a challenge without camera matrix & depth estimation.
- Challenges in traditional computer vision solutions are overcome by OverProjNet.
- OverProjNet infers latent relationships between 2D image plane and 3D scene space.
- Developed simulation tool can generate 2D pixel coordinates of overhead objects.
- Datasets are produced and released to validate the effectiveness of OverProjNet.

<p align="left"><img src="prints/teaser_1.png" width="1000"/></p>

## Abstract
Despite the availability of preventive and protective systems, accidents involving falling overhead objects, 
particularly load-bearing cranes, still occur and can lead to severe injuries or even fatalities. Therefore, it has 
become crucial to locate the projection of heavy overhead objects to alert those beneath and prevent such incidents. 
However, developing a generalized projection detector capable of handling various overhead objects with different 
sizes and shapes is a significant challenge. To tackle this challenge, we propose a novel approach called 
**OverProjNet**, which uses camera frames to visualize the overhead objects and the ground-level surface for 
projection detection. **OverProjNet** is designed to work with various overhead objects and cameras without any 
location or rotation constraints. To facilitate the design, development, and testing of **OverProjNet**, we provide 
two datasets: _CraneIntenseye_ and _OverheadSimIntenseye_. _CraneIntenseye_ comprises actual facility images, 
positional data of the overhead objects, and their corresponding predictions, while _OverheadSimIntenseye_ contains 
simulation data with similar content but generated using our simulation tool. Overall, **OverProjNet** achieves high 
detection performance on both datasets. The proposed solution's source code and our novel simulation tool are 
available at https://github.com/intenseye/overhead_object_projector. For the dataset and model zoo, please email the 
authors requesting access at https://drive.google.com/drive/folders/1to-5ND7xZaYojZs1aoahvu6BkLlYxRHP?usp=sharing.

<p align="left"><img src="prints/teaser_2.png" width="1000"/></p>


## Installation
```
conda create -n overhead_object_projector python=3.8
conda activate overhead_object_projector
sh install.sh 
```

## Dataset

- Download the dataset from https://drive.google.com/drive/folders/1to-5ND7xZaYojZs1aoahvu6BkLlYxRHP?usp=sharing
and place them in the main repository directory, as follows:

    ```
    [main_repo_folder]
       |——————Datasets
       |        └——————CraneIntenseye
       |        |         └——————Set01
       |        |         └——————Set02
       |        |         └——————ReadMe.txt
       |        └——————OverheadSimIntenseye
       |                  └——————Set01
       |                            └——————bbox_dev
       |                            └——————both_dev
       |                            └——————no_dev
       |                            └——————proj_dev
       |                  └——————Set02
       |                            └——————bbox_dev
       |                            └——————both_dev
       |                            └——————no_dev
       |                            └——————proj_dev
       |                  └——————Set03
       |                            └——————bbox_dev
       |                            └——————both_dev
       |                            └——————no_dev
       |                            └——————proj_dev
       |                  └——————Set04
       |                            └——————bbox_dev
       |                            └——————both_dev
       |                            └——————no_dev
       |                            └——————proj_dev
       |                  └——————Set05
       |                            └——————bbox_dev
       |                            └——————both_dev
       |                            └——————no_dev
       |                            └——————proj_dev
    ```

- While _OverheadSimIntenseye_ dataset is collected from a simulation environment, _CraneIntenseye_ dataset is collected 
from actual facility cameras.
- Both datasets comprise positional and visual data that indicate the pixel location of overhead objects as inputs and 
the center projection point of the overhead object as targets. While overhead objects are represented by bounding 
boxes, the center projection points of the overhead object are denoted by points. 

### OverheadSimIntenseye
- The _OverheadSimIntenseye_ comprises five distinct sets, each encompassing different camera placements, camera 
rotations, and variable sized objects, along with other camera and lens parameters. The parameters and their allowed 
ranges are presented in Table 1 in [Overhead object projector: OverProjNet](https://www.sciencedirect.com/science/article/pii/S2667305323000947) 
to generate each set of _OverheadSimIntenseye_. Simulation images are also provided for visual demonstrations and 
investigations, along with positional data.


- Bounding box boundaries and projection points are manipulated by adding random deviations within predefined limits. To 
analyze the effect of these deviations, we generate sets with and without applied deviations to the edges of the 
bounding boxes and the projection points of the objects (i.e., bbox_dev, proj_dev, both_dev, no_dev). 
  - bbox_dev: Deviations applied to bounding boxes
  - proj_dev: Deviations applied to projection points
  - both_dev: Deviations applied to both bounding boxes and projection points
  - no_dev: No deviations are applied. 


- In addition to the provided dataset, this repository offers the capability to create new simulation-based datasets. 
For a detailed explanation, refer to the **Data Sample Generation** section.

### CraneIntenseye
- The _CraneIntenseye_ dataset consists of two sets, where the inputs are obtained from actual cameras on facilities.The 
images are captured from fixed-position cameras while the cranes are in operation. The cameras used for data 
collection have different lens properties, viewpoints, and rotational angles, and different overhead objects (cranes)
are used during the collection of the _CraneIntenseye_ dataset.


- Since most of the parameters used to generate the sets of _OverheadSimIntenseye_ are unknown for _CraneIntenseye_, only 
the available ones are presented in Table 2 in [Overhead object projector: OverProjNet](https://www.sciencedirect.com/science/article/pii/S2667305323000947) 
for _CraneIntenseye_.

## Projection Trainer and Tester

#### Usage

- Run the following command to train and test projection point estimator (OverProjNet). Please see the _settings.ini_ 
files for the explanation of the projection trainer parameters. Also, please note that parameters swept during the 
hyperparameter search and parameters varying depending on network size are stored in _temp_params_ files. When 
network size is selected, the tuned hyperparameters will be loaded automatically.

    ```
    python projection_trainer.py --settings_path /path/to/the/settings.ini/file --network_size [network_size]
    ```
- _projection_trainer.py_ is responsible for training the OverProjNet model across a defined number of epochs. It 
monitors both loss and accuracy on both training and validation sets during the training process.
- _projection_trainer.py_ stores the best state of OverProjNet model by considering the best validation accuracy and 
utilize it for testing stages.
- _projection_trainer.py_ also evaluates the OverProjNet model's accuracy using a test dataset, calculating key metrics 
like loss, accuracy, and error.
- If the distance map in metric space is generated, the _sample_generator.py_ assesses accuracy and errors in metric 
distance space in addition to pixel space.
- _projection_trainer.py_ also displays images and object positions when drawing or demo modes are enabled via 
_settings.ini_ file.

## Data Sample Generation
- Data sample generation ability is used to generate data samples that include the position of an overhead object and 
its projection point in an image. It is designed to simulate scenarios where an object is observed by a camera and its 
location and projection point are recorded. The script includes functionalities for camera projection, object 
transformation, and adding random deviations to the data.

#### Usage

- Run the following commands to generate simulation samples. Please see the settings file for the explanation of the 
sample generation parameters.

    ```
    python sample_generator.py --settings_path /path/to/the/settings.ini/file
    ```

    i.e.,
    ```
    python sample_generator.py --settings_path ./settings/settings_1.ini
    ```

- _sample_generator.py_ generates data samples by varying object position, rotation, and applying random deviations 
depending on the set parameters.
- _sample_generator.py_ also optionally exports the data samples, including images and coordinates, to an output folder.
- _sample_generator.py_ also displays images and object positions when drawing or demo modes are enabled.


- After generating the samples, to work with the current functionalities of this repository, it is advice to split the 
sets into training, validation, and test subsets. Please see **Data Splitting Script** explanations under **Helper 
Abilities** section. If it is desired to use only a small portion of the generated samples to save disk space 
**Delete Unused Simulation Data Script** can be used (see **Helper Abilities** section). 

## Helper Abilities

### Data Splitting Script

- The Data Splitting Script is used to split a dataset into training, validation, and test subsets with the given ratios.

#### Usage

- Run the following command to split your dataset:

    ```
    python ./helper_scripts/split_simulation_datasets.py --data_folder_path /path/to/your/dataset --val_ratio [val_ratio] --test_ratio [test_ratio] --train_ratio [train_ratio]
    ```
    
    e.g.,
    ```
    python ./helper_scripts/split_simulation_datasets.py --data_folder_path ./datasets/OverheadSimIntenseye/Set01/2023_10_25_08_51_40_329 --val_ratio 0.01 --test_ratio 0.01 --train_ratio 0.004
    ```

- _split_simulation_datasets.py_ will create a new directory named "split" within your dataset folders (for each of 
bbox_dev, proj_dev, both_dev, no_dev) and organize the data into three JSON files: _coordinates_train.json_, 
_coordinates_val.json_, and _coordinates_test.json_.

### Delete Unused Simulation Data Script

- The Delete Unused Simulation Data Script is used to clean a dataset by removing unused data and associated JSON fields 
in any of training, validation and test split. This is helpful when you want to reduce the size of your dataset by 
removing unnecessary files and metadata.

#### Usage

- Run the following command to delete unused simulation data:

    ```
    python helper_scripts/delete_unused_simulation_data.py --data_folder_path /path/to/your/dataset
    ```
    
    e.g.,
    ```
    python helper_scripts/delete_unused_simulation_data.py --data_folder_path ./datasets/OverheadSimIntenseye/Set01/2023_10_25_08_51_40_329
    ```

- _delete_unused_simulation_data.py_ will scan the specified dataset folder and remove unused images and their 
associated metadata. It will help you clean your dataset by deleting unnecessary files.

## Time and Throughput Measurement Script

- The Time and Throughput Measurement ability is used to measure the processing time and throughput of a deep learning 
model implemented in PyTorch. This script is particularly useful when you want to assess the performance of a model on 
different network sizes and batch sizes.

#### Usage

- Run the script without any additional arguments. 

    ```
    python measure_time_throughput.py
    ```

- _measure_time_throughput.py_ perform time and throughput measurements for various network sizes and batch sizes.




