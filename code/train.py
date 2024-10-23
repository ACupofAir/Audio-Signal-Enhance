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
from datetime import datetime

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration parameters
config = {
    "task": "enh_single",
    "batch_size": 2,
    "sample_rate": 20000,
    "n_src": 1,
    "segment": 0.3,
    "learning_rate": 1e-3,
    "max_epochs": 30,
    "model_save_dir": r"C:\Users\june\Workspace\Asteroid\checkpoint",
    "example_wav_path": r"C:\Users\june\Workspace\Asteroid\code\example.wav",
}


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
    csv_dir = ""

    def __init__(
        self,
        csv_dir,
        task="enh_both",
        sample_rate=20000,
        n_src=2,
        segment=0.3,
        return_id=False,
    ):
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        # Get the csv corresponding to the task
        if task == "enh_single":
            md_file = [f for f in os.listdir(csv_dir) if "single" in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        elif task == "enh_both":
            self.csv_path = r"C:\Users\june\Workspace\Asteroid\data\metadata\train\mixture_train_mix_both.csv"
            md_clean_file = r"C:\Users\june\Workspace\Asteroid\data\metadata\train\mixture_train_mix_clean.csv"
            self.df_clean = pd.read_csv(md_clean_file)
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
        meta_path = cls.csv_dir
        # Create dataset instances
        train_set = cls(
            os.path.join(meta_path, "train"), sample_rate=125000, segment=0.3, **kwargs
        )
        val_set = cls(
            os.path.join(meta_path, "val"), sample_rate=125000, segment=0.3, **kwargs
        )
        return train_set, val_set


SanyaEnhance = Sanya(
    csv_dir=r"C:\Users\june\Workspace\Asteroid\data\metadata",
    task="enh_both",
    sample_rate=20000,
    n_src=2,
    segment=0.3,
)
train_loader, val_loader = SanyaEnhance.loaders_from_mini(task="enh_both", batch_size=2)
# train_loader, val_loader = Sanya.loaders_from_mini(task="enh_both", batch_size=2)

# Move model to the GPU
model = DPRNNTasNet(n_src=1, sample_rate=20000).to(device)
# model = asteroid.DPTNet(n_src=2, sample_rate=125000).to(device)
# PITLossWrapper works with any loss function.
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx").to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

system = System(model, optimizer, loss, train_loader, val_loader)

# trainer = Trainer(max_epochs=1, gpus=1)
trainer = Trainer(max_epochs=30)  # Specify to use 1 GPU
trainer.fit(system)

# model_save_path = 'DPTN-bs8_epoch50_sr125000.pth'
# Generate model name based on parameters
model_name = f"DPTN-bs{train_loader.batch_size}_epoch{trainer.max_epochs}_sr{train_loader.dataset.sample_rate}.pth"

# Create directory with current date
current_date = datetime.now().strftime("%Y-%m-%d")
model_dir = os.path.join(r"C:\Users\june\Workspace\Asteroid\checkpoint", current_date)
os.makedirs(model_dir, exist_ok=True)

# Full model save path
model_save_path = os.path.join(model_dir, model_name)

# Save the model state dict
torch.save(system.model, model_save_path)
model.separate(
    r"C:\Users\june\Workspace\Asteroid\code\example.wav",
    resample=True,
)
