# ConvLSTM for radar-based precipitation nowcasting

This repository contains the source code of a ConvLSTM encoder-forecaster used for radar-based precipitation nowcasting: given 5 observed radar frames, the model predicts the next 6 frames.

The repository structure follows the one used in _An Empirical Study Using Lightweight Convolutions to Achieve Green AI_ (STMixGAN baseline). This implementation uses standard PyTorch convolutions throughout; lightweight convolution variants are not used.

The datasets used are TAASRAD19 and MeteoNet (NW region), both already pre-windowed in `.npy` files with shape `(samples, 11, 1, 120, 120)`. The first 5 steps are the input and the last 6 steps are the target.

## Installing the project dependencies

The project was developed in **Python 3.11+**.

To install the dependencies, create a Python virtual environment of your choice. On Windows with an NVIDIA GPU, install the CUDA build of PyTorch first, so the next step does not replace it with a CPU build:

```shell
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

Then, run:

```shell
make setup
```

If you want to install the dependencies related to the project's development, run:

```shell
make setup-dev
```

Currently, the only development dependency is a Python code formatter, used to enforce PEP 8 style guidelines.

To apply the formatter, run:

```shell
make format
```

To install the dependencies required to run the experiments, run:

```shell
make setup-experiment
```

## Data

Place the `.npy` files in the dataset folders:

```
data
├── taasrad19/
│   ├── taasrad19_2010_2011_2012_2013_2014_2015_2016_120x120.npy   (train)
│   ├── taasrad19_2017_120x120.npy                                 (val)
│   └── taasrad19_2018_2019_120x120.npy                            (test)
└── meteonet/
    ├── meteonet_NW_2016_2017_120x120_clean.npy                    (train)
    ├── meteonet_NW_2017_limit_09_120x120_clean.npy                (val)
    └── meteonet_NW_2018_120x120_clean.npy                         (test)
```

The files are read in memory-mapped mode, so only the requested samples are loaded into RAM.

With `--normalize_values`, pixel values are divided by the dataset maximum (`max_value` in `datasets/taasrad19.py` and `datasets/meteonet.py`) to fall in the range [0, 1]. Confirm that these constants match your data.

## How to run the project

We have two executable scripts: `trainer.py` and `predictor.py`.
For more details on all available parameters to run the scripts above, execute one of the commands below, depending on the desired step:

```shell
python trainer.py --help
python predictor.py --help
```


### Training

To run a training process, use the following command:

```shell
python trainer.py \
    --num_epochs 4 \
    --batch_size 5 \
    --working_dir my_space/save_here \
    --start_epoch_saving 3 \
    --normalize_values \
    --model_architecture ms_convlstm \
    --dataset taasrad19 \
    --result_folder_suffix my_suffix
```

- num_epochs: Number of training epochs
- batch_size: Batch size for each training step
- working_dir: Path to the directory where training outputs will be saved
- start_epoch_saving: Initial epoch from which the model weights will be saved
- normalize_values: Normalize data values to the range [0, 1] before training
- model_architecture: Choose `convlstm` (default) or `ms_convlstm`. Both architectures use standard PyTorch convolutions.
- dataset: Name that identifies the dataset to be used. Options: taasrad19 and meteonet
- result_folder_suffix: Suffix name to be appended to the output folder created by the project. If not specified, the suffix will be the execution start datetime, in the format `YYYYMMDD_HHMMSS`

Other useful parameters: `--hidden_dims` (channels of each ConvLSTM layer, default `64 64`), `--lr`, `--teacher_forcing_ratio` (initial probability, decays linearly to 0 along the epochs), `--grad_clip`, `--use_seed`/`--seed` and `--check_model_summary`.

Both architectures use standard `torch.nn.Conv2d` operations; lightweight convolution selection is not supported. The `ms_convlstm` architecture follows the full/half/quarter/quarter/half/full spatial path from [MS_ConvLSTM.py](./MS_ConvLSTM.py); the last value in `--hidden_dims` is used as the channel count at every scale. Pass `--model_architecture ms_convlstm` to both `trainer.py` and `predictor.py` to select it. MS-ConvLSTM runs are saved under a distinct result-folder suffix, so they do not overwrite default ConvLSTM outputs.

After training, the following file structure will be generated:

```
my_space
├──save_here
│    ├──training_taasrad19_standard_ms_convlstm_my_suffix/
│    │    ├── emissions/
│    │    │ └── emissions.csv
│    │    ├── metrics/
│    │    │ ├── execution_metrics_training_standard_taasrad19.csv
│    │    │ ├── loss_epoch_standard_taasrad19.csv
│    │    │ └── loss_step_standard_taasrad19.csv
│    │    ├── weights/
│    │    │ └── convlstm/
│    │    │   ├── taasrad19_convlstm_standard_3.pth
│    │    │   └── taasrad19_convlstm_standard_4.pth
```

Training folders follow `training_[dataset]_standard_[result_folder_suffix]`. For `ms_convlstm`, the architecture is prefixed to the suffix to keep its artifacts separate.
Inside it, three subfolders are created: CodeCarbon metrics (emissions), training and execution metrics (metrics), and the model weights saved per epoch (weights).

### Prediction (Inference)

To run a prediction, use the following command:

```shell
python predictor.py \
    --model_weight_file_path my_space/save_here/training_taasrad19_standard_ms_convlstm_my_suffix/weights/convlstm/taasrad19_convlstm_standard_4.pth \
    --working_dir my_space/save_here \
    --normalize_values \
    --model_architecture ms_convlstm \
    --dataset taasrad19 \
    --result_folder_suffix my_suffix
```

- model_weight_file_path: Path to the model weights file used to load the model
- working_dir: Path to the directory where prediction outputs will be saved
- normalize_values: Normalize data values to the range [0, 1] before prediction
- model_architecture: Set to `ms_convlstm` when loading a checkpoint trained with the multi-scale architecture; it defaults to `convlstm`.
- dataset: Name that identifies the dataset to be used. Options: taasrad19 and meteonet
- result_folder_suffix: Suffix name to be appended to the output folder created by the project. If not specified, the suffix will be the execution start datetime, in the format `YYYYMMDD_HHMMSS`

The selected architecture and `--hidden_dims` must match the training run, otherwise the weights cannot be loaded.

To evaluate a checkpoint on the validation split instead of the test split, pass `--dataset_mode val`.

After prediction, the following file structure will be generated:

```
my_space
├──save_here
│    ├──predict_taasrad19_standard_ms_convlstm_my_suffix/
│    │    ├── emissions/
│    │    │   └── emissions.csv
│    │    ├── images/
│    │    │   ├── generated/
│    │    │   └── real/
│    │    ├── metrics/
│    │    │   ├── execution_metrics_predict_standard_taasrad19.csv
│    │    │   ├── standard_taasrad19_fss.csv
│    │    │   ├── standard_taasrad19_image_metrics.csv
│    │    │   ├── standard_taasrad19_pooled_crps.csv
│    │    │   └── standard_taasrad19_threshold_metrics.csv
│    │    ├── npy_files/
│    │    │   ├── generated/
│    │    │   └── real/
```

Prediction folders follow `predict_[dataset]_standard_[result_folder_suffix]`. For `ms_convlstm`, the architecture is prefixed to the suffix.

Inside it, four subfolders are created: CodeCarbon metrics (`emissions`), predictive performance and execution metrics (`metrics`), images in .png format of the real and generated sequences (`images`), and a folder containing the real and generated images in .npy format (`npy_files`).

The images and npy_files folders are generated optionally. To enable their creation in addition to the other folders, you must pass the flags `--save_sequence_images` and `--save_sequence_npy_files`. This design choice was made to save disk space for the files generated during prediction.

If we want to skip the evaluation step in the prediction process, we can pass the flag `--skip_run_evaluation`.


## Reproducing the experiments with IOPS

The dataset runs are orchestrated with [IOPS](https://iops-benchmark.com), a benchmark orchestration framework. The provided configurations sweep both `convlstm` and `ms_convlstm`, as well as the configured datasets, and repeat runs to collect comparable metrics. The IOPS `repetitions` replace the multiple seeds.

Install IOPS with:

```shell
pip install iops-benchmark
```

The `requirements-experiment.txt` file contains the dependencies required to run the experiments with IOPS. If you have already installed the dependencies with `make setup-experiment`, you can skip installing the IOPS package, as it is included in the `requirements-experiment.txt` file.

The GPU usage probe relies on `nvidia-smi`, so an NVIDIA GPU with the NVIDIA drivers installed is required for the `gpu_sampling` metrics. Make sure `nvidia-smi` is available on the compute nodes.

Three IOPS configuration files are provided:

- `iops_train.yaml`: sweeps datasets and architectures to run the training step (`trainer.py`).
- `iops_predict.yaml`: sweeps datasets and architectures to predict using the corresponding trained weights.
- `iops_predict_evaluation.yaml`: sweeps datasets and architectures to predict using the corresponding trained weights and evaluates the results.

Before running, adjust `project_dir` in each file to point to the directory holding this code, and review the SLURM directives in `script_template` (node list, partition, modules) to match your cluster.

The SLURM scripts use a Python virtual environment at `<project_dir>/.venv` (the `venv_dir` variable in each YAML). Create it on a compatible Linux environment that is accessible from the compute nodes, and install both the project and experiment dependencies:

```shell
cd /home/<some-user>/convlstm-green-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# First install the PyTorch/torchvision build compatible with the cluster's CUDA setup.
python -m pip install -r requirements.txt -r requirements-experiment.txt
```

Use a PyTorch build compatible with the cluster's GPU and CUDA driver when preparing the environment. The jobs call `.venv/bin/python` directly and fail with a clear message if it is missing. On the login node, activate the same environment before invoking IOPS so the `iops` command is available:

```shell
source .venv/bin/activate
```

To launch the experiments:

```shell
iops run iops_train.yaml
iops run iops_predict.yaml
iops run iops_predict_evaluation.yaml
```

See the documentation at [iops-benchmark.com](https://iops-benchmark.com) for more details.

### Get IOPS metrics

When running IOPS experiments, two output files are generated in the experiment's root directory: `results.csv` and `__iops_resource_summary.csv`.

- results.csv (iops_train.yaml):
  - **metrics.total_training_time_sec**: Execution time in seconds
  - **metadata.sysinfo.gpu_memory_mib**: GPU total memory in MiB
- results.csv (iops_predict.yaml):
  - **metrics.total_predict_time_sec**: Execution time in seconds
  - **metadata.sysinfo.gpu_memory_mib**: GPU total memory in MiB
- results.csv (iops_predict_evaluation.yaml):
  - **metrics.MAE**
  - **metrics.RMSE**
  - **metrics.SSIM**
- __iops_resource_summary.csv:
  - **gpu0_energy_j**: GPU energy consumption in Joules
  - **gpu0_mem_peak_mib**: GPU peak memory usage in MiB
  - **gpu0_avg_mem_utilization_pct**: GPU memory utilization percentage

To obtain the average GPU memory usage over the entire execution:

$$
avg\_mem\_uti = \frac{gpu0\_avg\_mem\_utilization\_pct}{100} * gpu\_memory\_mib
$$

To convert in kWh:

$$
kWh = \frac{gpu0\_energy\_j}{3600000}
$$

## Project structure

```
.
├── ConvLSTM.py                  # ConvLSTM cell and encoder-forecaster model
├── argsparser.py                # CLI arguments for trainer.py and predictor.py
├── data_manipulation.py         # dataset selection, loading and denormalization
├── datasets/                    # TAASRAD19 and MeteoNet dataset classes
├── ConvLSTM.py                  # ConvLSTM encoder-forecaster
├── MS_ConvLSTM.py               # multi-scale ConvLSTM encoder-forecaster
├── loss.py                      # sequence MSE loss
├── metrics.py                   # evaluation metrics
├── predictor.py                 # inference and evaluation
├── trainer.py                   # training
├── utils.py                     # output folders and sequence saving
└── iops_*.yaml                  # IOPS experiment definitions
```
