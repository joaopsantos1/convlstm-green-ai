import os
from time import perf_counter
from codecarbon import EmissionsTracker
import pandas as pd
import torch
from torchinfo import summary
from tqdm import tqdm
from argsparser import create_train_parser, validate_train_args
from data_manipulation import get_datasets_parameters, load_dataset
from loss import SequenceMSELoss
from MS_ConvLSTM import MS_ConvLSTM
from utils import CONVOLUTION_NAME, create_training_working_directory


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.model_architecture = self.args.model_architecture
        self._setup_cuda_device()
        self._preparation()

    def _setup_cuda_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _preparation(self):
        self._init_random_seed()
        self._get_dataset_params()
        self._get_data()
        self._get_epoch()
        self._build_model()
        self._select_optimizer()
        self._select_loss()
        self._training_directory_setup()

    def _init_random_seed(self):
        # To make start weights values equal in any training execution.
        if self.args.use_seed:
            print(f"[INFO] Using {self.args.seed} for random seed...")
            torch.manual_seed(self.args.seed)

    def _training_directory_setup(self):
        self.train_dirs = create_training_working_directory(
            self.args.working_dir,
            self.args.dataset,
            self.args.result_folder_suffix,
            self.model_architecture,
        )

    def _build_model(self):
        if self.model_architecture == "ms_convlstm":
            self.model = MS_ConvLSTM(
                in_channels=1,
                output_seq_len=self.output_seq_len,
                hidden_dims=self.args.hidden_dims,
            )
        else:
            from ConvLSTM import ConvLSTMNet

            self.model = ConvLSTMNet(
                in_channels=1,
                output_seq_len=self.output_seq_len,
                hidden_dims=self.args.hidden_dims,
            )
        self.model.cuda()

    def _get_dataset_params(self):
        print(f"[INFO] Dataset used: {self.args.dataset}")
        dataset_params = get_datasets_parameters(self.args.dataset)
        self.input_seq_len = dataset_params["input_seq_len"]
        self.output_seq_len = dataset_params["output_seq_len"]
        self.input_size = dataset_params["input_size"]

    def _get_data(self):
        self.train_loader, _ = load_dataset(
            name=self.args.dataset,
            dataset_mode="train",
            dataset_size=self.args.dataset_size,
            normalize_values=self.args.normalize_values,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

    def _get_epoch(self):
        self.total_epochs = self.args.num_epochs
        self.train_len = len(self.train_loader)

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.weight_decay,
        )

    def _select_loss(self):
        self.loss_fn = SequenceMSELoss()

    def _teacher_forcing_ratio(self, epoch: int) -> float:
        # Linear decay from the initial ratio (first epoch) to 0 (last epoch).
        progress = epoch / max(self.total_epochs - 1, 1)
        return self.args.teacher_forcing_ratio * max(0.0, 1 - progress)

    def train(self):
        self.model.train()

        min_step_loss = 1_000_000
        min_epoch_loss = 1_000_000
        min_epoch = 0
        training_step_metrics = []
        training_epoch_metrics = []
        step_itr = 1

        with EmissionsTracker(
            project_name=(
                f"{'MS-ConvLSTM' if self.model_architecture == 'ms_convlstm' else 'ConvLSTM'}-"
                f"{self.args.dataset}-train-{CONVOLUTION_NAME}"
            ),
            log_level="error",
            output_dir=self.train_dirs["emissions"],
        ) as tracker:
            start = perf_counter()

            for epoch in range(self.total_epochs):
                teacher_forcing_ratio = self._teacher_forcing_ratio(epoch)
                epoch_loss_sum = 0.0
                epoch_samples = 0

                pbar = tqdm(
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}/{self.total_epochs}",
                    postfix=dict,
                    mininterval=0.3,
                )  # Setting the current epoch display progress

                for itr, train_imgs in enumerate(self.train_loader):
                    train_imgs = train_imgs.type(torch.cuda.FloatTensor)
                    train_input = train_imgs[:, : self.input_seq_len]
                    train_target = train_imgs[
                        :,
                        self.input_seq_len : (self.input_seq_len + self.output_seq_len),
                    ]

                    # -------------------------------------------------------------------------------------------------
                    # Train ConvLSTM
                    self.optimizer.zero_grad()

                    train_gen = self.model(
                        train_input,
                        target=train_target,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                    )

                    loss = self.loss_fn(train_gen, train_target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.grad_clip
                    )
                    self.optimizer.step()
                    # -------------------------------------------------------------------------------------------------

                    pbar.set_postfix(**{"loss": loss.item()})
                    pbar.update(1)

                    epoch_loss_sum += loss.item() * train_input.shape[0]
                    epoch_samples += train_input.shape[0]

                    if min_step_loss > loss.item():
                        min_step_loss = loss.item()

                    # Adding values to the metrics list
                    training_step_metrics.append(
                        {
                            "epoch": epoch + 1,
                            "step": step_itr,
                            "loss": loss.item(),
                        }
                    )
                    step_itr += 1

                pbar.close()  # Turn off the current epoch display progress

                epoch_loss = epoch_loss_sum / epoch_samples

                if min_epoch_loss > epoch_loss:
                    min_epoch_loss = epoch_loss
                    min_epoch = epoch + 1

                training_epoch_metrics.append(
                    {
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "teacher_forcing_ratio": teacher_forcing_ratio,
                        "min_step_loss": min_step_loss,
                        "min_until_epoch_loss": min_epoch_loss,
                    }
                )

                if epoch + 1 >= self.args.start_epoch_saving:
                    torch.save(
                        self.model.state_dict(),
                        f"{self.train_dirs['weights']['model']}/{self.args.dataset}_{self.model_architecture}_{CONVOLUTION_NAME}_{epoch+1}.pth",
                    )

            end = perf_counter()
        print(
            "Training Time Consumption：{:.0f}min {:.1f}s".format(
                (end - start) // 60, (end - start) % 60
            )
        )
        print(f"Training minimum epoch loss: {min_epoch_loss}")
        print(f"Training minimum epoch: {min_epoch}")

        training_execution_metrics = [
            {
                "total_training_time_sec": (end - start),
                "total_training_time_part_hour": (end - start) // 3600,
                "total_training_time_part_min": ((end - start) % 3600) // 60,
                "total_training_time_part_sec": (end - start) % 60,
                "average_epoch_time_sec": (end - start) / self.args.num_epochs,
                "min_epoch_loss": min_epoch_loss,
                "min_epoch": min_epoch,
            }
        ]
        df_execution = pd.DataFrame(training_execution_metrics)
        df_execution.to_csv(
            f"{self.train_dirs['metrics']}/execution_metrics_training_{CONVOLUTION_NAME}_{self.args.dataset}.csv",
            index=False,
        )

        df_step = pd.DataFrame(training_step_metrics)
        df_step.to_csv(
            f"{self.train_dirs['metrics']}/loss_step_{CONVOLUTION_NAME}_{self.args.dataset}.csv",
            index=False,
        )
        df_epoch = pd.DataFrame(training_epoch_metrics)
        df_epoch.to_csv(
            f"{self.train_dirs['metrics']}/loss_epoch_{CONVOLUTION_NAME}_{self.args.dataset}.csv",
            index=False,
        )


if __name__ == "__main__":
    args = create_train_parser().parse_args()
    validate_train_args(args)

    trainer = Trainer(args)

    if args.check_model_summary:
        summary(
            trainer.model,
            input_size=(
                args.batch_size,
                trainer.input_seq_len,
                1,
                trainer.input_size,
                trainer.input_size,
            ),
        )
    else:
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>Start of training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        trainer.train()
