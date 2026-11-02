import os
from time import perf_counter
from codecarbon import EmissionsTracker
import pandas as pd
import torch
from tqdm import tqdm
from argsparser import create_predict_parser
from data_manipulation import get_datasets_parameters, load_dataset, denormalize_values
from metrics import Evaluator
from MS_ConvLSTM import MS_ConvLSTM
from utils import (
    CONVOLUTION_NAME,
    create_predict_working_directory,
    save_sequence_images,
    save_sequence_npy,
)


class Predictor:
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self.model_architecture = self.args.model_architecture
        self._setup_cuda_device()
        self._preparation()

    def _setup_cuda_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _preparation(self):
        self._get_dataset_params()
        self._get_data()
        self._predict_directory_setup()
        self._get_evaluator()
        self._build_model()

    def _predict_directory_setup(self):
        self.predict_dirs = create_predict_working_directory(
            self.args.working_dir,
            self.args.dataset,
            self.args.result_folder_suffix,
            self.model_architecture,
        )

    def _get_data(self):
        self.test_loader, self.dataset_max_value = load_dataset(
            name=self.args.dataset,
            dataset_mode=self.args.dataset_mode,
            dataset_size=self.args.dataset_size,
            normalize_values=self.args.normalize_values,
            batch_size=1,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def _get_evaluator(self):
        self.evaluator = Evaluator(
            model_name=CONVOLUTION_NAME,
            dataset=self.args.dataset,
            output_seq_len=self.output_seq_len,
            metric_dir=self.predict_dirs["metrics"],
            pixel_thresholds=self.args.pixel_thresholds,
            pixel_balancing_weights=self.args.pixel_balancing_weights,
            dataset_max_value=self.dataset_max_value
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
        self.model.load_state_dict(
            torch.load(self.args.model_weight_file_path, weights_only=True)
        )
        self.model.cuda()

    def _get_dataset_params(self):
        print(f"[INFO] Dataset used: {self.args.dataset}")
        dataset_params = get_datasets_parameters(self.args.dataset)
        self.input_seq_len = dataset_params["input_seq_len"]
        self.output_seq_len = dataset_params["output_seq_len"]
        self.input_size = dataset_params["input_size"]

    def predict(self):
        self.model.eval()

        with torch.no_grad():
            output = []
            groundtruth = []

            with EmissionsTracker(
                project_name=(
                    f"{'MS-ConvLSTM' if self.model_architecture == 'ms_convlstm' else 'ConvLSTM'}-"
                    f"{self.args.dataset}-predict-{CONVOLUTION_NAME}"
                ),
                log_level="error",
                output_dir=self.predict_dirs["emissions"],
            ) as tracker:
                start = perf_counter()

                for step, test_imgs in enumerate(tqdm(self.test_loader, desc="Steps")):
                    test_imgs = test_imgs.type(torch.cuda.FloatTensor)

                    input = test_imgs[:, : self.input_seq_len]
                    target = test_imgs[
                        :,
                        self.input_seq_len : (self.input_seq_len + self.output_seq_len),
                    ]

                    pred_imgs_cat = self.model(input)

                    groundtruth.append(target)
                    output.append(pred_imgs_cat)

                    real_seq = torch.cat([input, target], dim=1)
                    pred_seq = torch.cat([input, pred_imgs_cat], dim=1)

                    if self.args.save_sequence_npy_files:
                        save_sequence_npy(
                            batch_seq_images=real_seq,
                            saving_dir=self.predict_dirs["npy_files"]["real"],
                            step=step + 1,
                        )

                        save_sequence_npy(
                            batch_seq_images=pred_seq,
                            saving_dir=self.predict_dirs["npy_files"]["generated"],
                            step=step + 1,
                        )

                    if self.args.save_sequence_images:
                        save_sequence_images(
                            batch_seq_images=torch.cat([input, target], dim=1),
                            batch_size=input.shape[0],
                            saving_dir=self.predict_dirs["images"]["real"],
                            step=step + 1,
                        )

                        save_sequence_images(
                            batch_seq_images=torch.cat([input, pred_imgs_cat], dim=1),
                            batch_size=input.shape[0],
                            saving_dir=self.predict_dirs["images"]["generated"],
                            step=step + 1,
                        )

                groundtruth = torch.cat(groundtruth, dim=0)
                pred_imgs = torch.cat(output, dim=0)

                end = perf_counter()

        if not self.args.skip_run_evaluation:
            # To correctly calculate some metrics (like MAE, MSE and their balanced ones) we need to use the original value interval.
            if self.args.normalize_values:
                groundtruth = denormalize_values(
                    groundtruth, max_value=self.dataset_max_value
                )
                pred_imgs = denormalize_values(pred_imgs, max_value=self.dataset_max_value)

            self.evaluator.save_image_metrics(groundtruth, pred_imgs)
            self.evaluator.save_score_csv(groundtruth, pred_imgs)
            print("Saved all metrics in CSVs files!")
            end_with_evaluation = perf_counter()
        else:
            end_with_evaluation = end

        print(
            "Forecasting Time Consumption：{:.0f}min {:.1f}s".format(
                (end - start) // 60, (end - start) % 60
            )
        )

        predict_execution_metrics = [
            {
                "total_predict_time_sec": (end - start),
                "total_predict_time_with_evaluation_sec": (end_with_evaluation - start),
                "total_predict_time_part_hour": (end - start) // 3600,
                "total_predict_time_part_min": ((end - start) % 3600) // 60,
                "total_predict_time_part_sec": (end - start) % 60,
            }
        ]
        df_execution = pd.DataFrame(predict_execution_metrics)
        df_execution.to_csv(
            f"{self.predict_dirs['metrics']}/execution_metrics_predict_{CONVOLUTION_NAME}_{self.args.dataset}.csv",
            index=False,
        )


if __name__ == "__main__":
    args = create_predict_parser().parse_args()

    predictor = Predictor(args)
    print(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>Start of forecasting<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )
    predictor.predict()
