import numpy as np
import pandas as pd
import soundfile as sf
import torch
from asteroid.engine import System
from torch import hub
from torch.utils.data import Dataset, DataLoader
import random as random
import os
import shutil
import zipfile
from torch import optim
from pytorch_lightning import Trainer
import pytorch_lightning as pl

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet
import asteroid.models.dptnet
# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Sanya(Dataset):
    """Dataset class for LibriMix source separation tasks.

    Args:
        csv_dir (str): The path to the metadata file.
        task (str): One of ``'enh_single'``, ``'enh_both'``, ``'sep_clean'`` or
            ``'sep_noisy'`` :

            * ``'enh_single'`` for single speaker speech enhancement.
            * ``'enh_both'`` for multi speaker speech enhancement.
            * ``'sep_clean'`` for two-speaker clean source separation.
            * ``'sep_noisy'`` for two-speaker noisy source separation.

        sample_rate (int) : The sample rate of the sources and mixtures.
        n_src (int) : The number of sources in the mixture.
        segment (int, optional) : The desired sources and mixtures length in s.

    References
        [1] "LibriMix: An Open-Source Dataset for Generalizable Speech Separation",
        Cosentino et al. 2020.
    """

    dataset_name = "Sanya"

    def __init__(
            self, csv_dir, task="sep_clean", sample_rate=20000, n_src=1, segment=0.3, return_id=False
    ):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
            md_clean_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.df_clean = pd.read_csv(os.path.join(csv_dir, md_clean_file))
        elif task == "sep_clean":
            md_file = [f for f in os.listdir(csv_dir) if "clean" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "sep_noisy":
            md_file = [f for f in os.listdir(csv_dir) if "both" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture
        if "enh_both" in self.task:
            mix_clean_path = self.df_clean.iloc[idx]["mixture_path"]
            s, _ = sf.read(mix_clean_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)

        else:
            # Read sources
            for i in range(self.n_src):
                source_path = row[f"source_{i + 1}_path"]
                s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
                sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        # 5400-34479-0005_4973-24515-0007.wav
        id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
        return mixture, sources, [id1, id2]

    @classmethod
    def loaders_from_mini(cls, batch_size=4, **kwargs):

        train_set, val_set = cls.mini_from_download(**kwargs)
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        return train_loader, val_loader

    @classmethod
    def mini_from_download(cls, **kwargs):
        # kwargs checks
        assert "csv_dir" not in kwargs, "Cannot specify csv_dir when downloading."
        assert kwargs.get("task", "sep_clean") in [
            "sep_clean",
            "sep_noisy",
        ], "Only clean and noisy separation are supported in MiniLibriMix."
        assert (
                kwargs.get("sample_rate", 125000) == 125000
        ), "Only 8kHz sample rate is supported in MiniLibriMix."

        meta_path = "../metadata"
        # Create dataset instances
        train_set = cls(os.path.join(meta_path, "train"), sample_rate=125000,segment=0.3, **kwargs)
        val_set = cls(os.path.join(meta_path, "val"), sample_rate=125000,segment=0.3, **kwargs)
        return train_set, val_set


train_loader, val_loader = Sanya.loaders_from_mini(task="enh_single", batch_size=2)

# Move model to the GPU
model = DPRNNTasNet(n_src=1, sample_rate=20000).to(device)
#model = asteroid.DPTNet(n_src=2, sample_rate=125000).to(device)
# PITLossWrapper works with any loss function.
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx").to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

system = System(model, optimizer, loss, train_loader, val_loader)

# trainer = Trainer(max_epochs=1, gpus=1)
trainer = Trainer(max_epochs=30)  # Specify to use 1 GPU
trainer.fit(system)

#model_save_path = 'DPTN-bs8_epoch50_sr125000.pth'
model_save_path = 'DPRNN-bs2_epoch30_sr125000.pth'

# Save the model state dict
torch.save(system.model, model_save_path)
model.separate("Data 1 Target1No 1Cycle No 1Group Depth500m_Data 2 Target2No 10Cycle No 6Group Depth1200m.wav", resample=True)
model.separate("Data 1 Target1No 1Cycle No 2Group Depth700m_Data 2 Target2No 8Cycle No 20Group Depth500m.wav", resample=True)
model.separate("Data 1 Target1No 1Cycle No 3Group Depth200m_Data 2 Target2No 4Cycle No 14Group Depth400m.wav", resample=True)