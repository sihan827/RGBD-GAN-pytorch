RGBD-GAN PyTorch Implementation with One Stage-GAN Algorithm
============================================================
A PyTorch Implementation of RGBD-GAN with One stage-GAN loss function based on paper [RGBD-GAN: Unsupervised 3D Representation Learning from Natural Image Datasets via RGBD Image Synthesis](https://arxiv.org/abs/1909.12573) and [Training Generative Adversarial Networks in One Stage](https://arxiv.org/abs/2103.00430)

Requirement
-----------
* Argparse
* Numpy
* Matplotlib
* Python 3.7
* PyTorch
* TorchVision
* tqdm
* yaml

How to train a model?
---------------------

### configuration
You can modify config/ffhq_pggan_test.yml to apply different hyperparameters and configure dataset path and training sample result path.

### Dataset
You can download Flickr-Faces-HQ Dataset from [official GitHub repository](https://github.com/NVlabs/ffhq-dataset). You have to modify dataset path in configuration.

### Training
Run main.py with Flickr-Faces-HQ Dataset.

```
$ python3 main.py --h
usage: main.py [-h] [--root ROOT] [--config_path CONFIG_PATH]

optional arguments:
  -h, --help                    show this help message and exit
  --root ROOT                   directory contains the data and outputs
  --config_path CONFIG_PATH     config file path
```
### Generating Samples
Run generate_sample.py to generate sample from model.

```
$ python3 generate_sample.py --h
usage: generate_sample.py [-h] --cp_path CP_PATH [--save_path SAVE_PATH] [--grid_mode GRID_MODE] [--y_rotate Y_ROTATE]

optional arguments:
  -h, --help                    show this help message and exit
  --cp_path CP_PATH             directory contains the generator checkpoint
  --save_path SAVE_PATH         directory sample saved
  --grid_mode GRID_MODE         if True, make a sample images grid with 4 samples. else, make each images with 1 sample
  --y_rotate Y_ROTATE           y_angle value to rotate samples

```

Test Results
------------
Left sample is from normal two stage based RGBD-GAN, and right sample is from one stage based RGBD-GAN.

![outputs](https://github.com/sihan827/RGBD-GAN-pytorch/blob/main/sample/sample.png)