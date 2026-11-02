from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets.validate_data import validate_loaded_sequence_length


class TAASRAD19(Dataset):
    metadata = {"input_seq_len": 5, "output_seq_len": 5, "input_size": 64}

    def __init__(
        self,
        dataset_mode="train",
        normalize: bool = False,
        transform: Optional[Callable] = None,
    ):
        if dataset_mode == "train":
            data_file_name = "taasrad19_2016_2017_2018_64x64.npy"
        elif dataset_mode == "test":
            data_file_name = "taasrad19_2019_64x64.npy"
        else:
            raise ValueError(
                f"Dataset mode '{dataset_mode}' is not available for TAASRAD19; "
                "supported modes are 'train' and 'test'."
            )

        folder_data = Path(__file__).resolve().parent.parent
        # Memory-mapped: only the requested samples are read from disk.
        self.data = np.load(
            folder_data / "data" / "taasrad19" / data_file_name, mmap_mode="r"
        )
        self.max_value = 52.5
        self.normalize = normalize

        validate_loaded_sequence_length(
            seq_len=self.data.shape[1],
            input_seq_len=self.metadata["input_seq_len"],
            output_seq_len=self.metadata["output_seq_len"],
        )

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = torch.from_numpy(np.asarray(self.data[idx], dtype=np.float32))
        if self.normalize:
            data = self.normalize_values(data)
        if self.transform is not None:
            data = self.transform(data)

        return data

    def normalize_values(self, values):
        return values / self.max_value

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
        max_value = dataset.max_value

        # Tests purposes
        if dataset_size < 1.0:
            num_samples = round(dataset_size * len(dataset))
            dataset = Subset(dataset, range(num_samples))

        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
        )

        return dataloader, max_value

