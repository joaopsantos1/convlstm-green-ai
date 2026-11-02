import argparse


def add_common_arguments(parser):
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="How many subprocesses to use for data loading. '0' means that the data will be loaded in the main process. "
        "This parameter will be use in Pytorch DataLoader class",
    )
    parser.add_argument(
        "--working_dir",
        default="./",
        type=str,
        help="Directory to store execution results",
    )
    parser.add_argument(
        "--dataset_size", default=1.0, type=float, help="Percentage of dataset to use"
    )
    parser.add_argument(
        "--dataset",
        default="moving_mnist",
        type=str,
        help="Name of dataset to use",
    )
    parser.add_argument(
        "--normalize_values",
        action="store_true",
        help="It will normalize the image pixels values between 0 and 1",
    )
    parser.add_argument(
        "--model_architecture",
        choices=["convlstm", "ms_convlstm"],
        default="convlstm",
        help="Recurrent architecture to use for training or prediction",
    )
    parser.add_argument(
        "--hidden_dims",
        default=[64, 64],
        type=int,
        nargs="+",
        help="Number of channels per ConvLSTM layer; MS-ConvLSTM uses the final value at every scale",
    )
    parser.add_argument(
        "--result_folder_suffix",
        default="",
        type=str,
        help="Suffix name of folder result. If not set, "
        "the execution datetime will be used.",
    )


def create_train_parser():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    parser.add_argument(
        "--batch_size", default=5, type=int, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="Number of epochs to train"
    )
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument(
        "--lr_beta1",
        default=0.9,
        type=float,
        help="Learning rate Beta 1 for Adam optimizer",
    )
    parser.add_argument(
        "--lr_beta2",
        default=0.999,
        type=float,
        help="Learning rate Beta 2 for Adam optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay for Adam optimizer",
    )
    parser.add_argument(
        "--grad_clip",
        default=1.0,
        type=float,
        help="Maximum gradient norm (gradient clipping)",
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        default=0.5,
        type=float,
        help="Initial teacher forcing probability. It decays linearly to 0 "
        "along the epochs",
    )
    parser.add_argument(
        "--check_model_summary",
        action="store_true",
        help="Shows models summary details using "
        "torchinfo instead of training the models",
    )
    parser.add_argument(
        "--start_epoch_saving",
        default=1,
        type=int,
        help="Epoch number to start saving model weights",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed number, to guarantee reproducibility",
    )
    parser.add_argument(
        "--use_seed",
        action="store_true",
        help="Use or not use a random seed number to get the same random weights",
    )

    return parser


def create_predict_parser():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    parser.add_argument(
        "--pixel_thresholds",
        default=[0.5, 2, 5, 10, 30],
        type=float,
        nargs="*",
        help="Pixel thresholds to use to calculate " "score metrics",
    )
    parser.add_argument(
        "--pixel_balancing_weights",
        default=[1, 1, 2, 5, 10, 30],
        type=float,
        nargs="*",
        help="Pixel balancing weights to use " "to calculate metrics",
    )
    parser.add_argument(
        "--model_weight_file_path",
        type=str,
        required=True,
        help="Path of the model weight file (.pth) saved by trainer.py",
    )
    parser.add_argument(
        "--save_sequence_images",
        action="store_true",
        help="The prediction will save the real and predicted sequences images",
    )
    parser.add_argument(
        "--save_sequence_npy_files",
        action="store_true",
        help="The prediction will save the real and predicted sequences .npy files",
    )
    parser.add_argument(
        "--dataset_mode",
        choices=["test", "val"],
        default="test",
        type=str,
        help="Dataset type to select in predict. " "Can be 'test' or 'val'",
    )

    parser.add_argument(
        "--skip_run_evaluation",
        action="store_true",
        help="Skip run model evaluation",
    )

    return parser


def validate_train_args(args):
    if args.num_epochs <= 0:
        raise argparse.ArgumentTypeError(
            f"'num_epochs' must be positive. Get value: {args.num_epochs}"
        )

    if args.num_epochs < args.start_epoch_saving:
        raise argparse.ArgumentTypeError(
            f"'num_epochs' (={args.num_epochs}) must be greater than or equal to 'start_epoch_saving' (={args.start_epoch_saving})"
        )
