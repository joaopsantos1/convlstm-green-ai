from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from datasets.validate_data import validate_loaded_sequence_length


class MeteoNet(Dataset):
    metadata = {"input_seq_len": 5, "output_seq_len": 5, "input_size": 64}
    max_value = 70.0
    data_files = {
        "train": "meteonet_NW_2016_2017_120x120_clean.npy",
        "val": "meteonet_NW_2017_limit_09_120x120_clean.npy",
        "test": "meteonet_NW_2018_120x120_clean.npy",
    }

    def __init__(
        self,
        dataset_mode="train",
        normalize: bool = False,
        transform: Optional[Callable] = None,
    ):
        try:
            data_file_name = self.data_files[dataset_mode]
        except KeyError:
            supported_modes = ", ".join(self.data_files)
            raise ValueError(
                f"Dataset mode '{dataset_mode}' is not available for MeteoNet; "
                f"supported modes are {supported_modes}."
            ) from None

        folder_data = Path(__file__).resolve().parent.parent
        self.data = np.load(
            folder_data / "data" / "meteonet" / data_file_name, mmap_mode="r"
        )
        self.normalize = normalize
        self.transform = transform

        validate_loaded_sequence_length(
            seq_len=self.data.shape[1],
            input_seq_len=self.metadata["input_seq_len"],
            output_seq_len=self.metadata["output_seq_len"],
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sequence_length = (
            self.metadata["input_seq_len"] + self.metadata["output_seq_len"]
        )
        data = torch.from_numpy(
            np.asarray(self.data[idx, :sequence_length], dtype=np.float32)
        )
        if data.shape[-2:] != (self.metadata["input_size"],) * 2:
            data = F.interpolate(
                data,
                size=(self.metadata["input_size"], self.metadata["input_size"]),
                mode="area",
            )
        if self.normalize:
            data = data / self.max_value
        if self.transform is not None:
            data = self.transform(data)

        return data

    @classmethod
    def load(
        cls,
        dataset_mode,
        dataset_size,
        normalize_values,
        batch_size,
        num_workers,
        shuffle=False,
    ):
        dataset = cls(
            dataset_mode=dataset_mode,
            normalize=normalize_values,
        )

        if dataset_size < 1.0:
            num_samples = round(dataset_size * len(dataset))
            dataset = Subset(dataset, range(num_samples))

        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )

        return dataloader, cls.max_value
