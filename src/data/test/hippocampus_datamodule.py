import tarfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"


def fit_size(a, size):
    """Pad (bên phải) hoặc crop (giữa) từng chiều về đúng kích thước size."""
    for ax, s in enumerate(size):
        n = a.shape[ax]
        if n < s:
            pad = [(0, 0)] * a.ndim
            pad[ax] = (0, s - n)
            a = np.pad(a, pad)
        elif n > s:
            start = (n - s) // 2
            a = np.take(a, range(start, start + s), axis=ax)
    return a


class VolDataset(Dataset):
    def __init__(self, pairs, size):
        self.pairs, self.size = pairs, size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp = self.pairs[i]
        img = nib.load(ip).get_fdata().astype(np.float32)
        lab = nib.load(lp).get_fdata().astype(np.int64)
        img = (img - img.mean()) / (img.std() + 1e-8)      # z-score từng volume
        img, lab = fit_size(img, self.size), fit_size(lab, self.size)
        return torch.from_numpy(img)[None], torch.from_numpy(lab)   # (1,D,H,W), (D,H,W)


class SegDataModule(LightningDataModule):
    def __init__(self, data_dir="data/", batch_size=2, num_workers=0,
                 pin_memory=False, patch_size=(48, 64, 48), val_split=0.2):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        root = Path(self.hparams.data_dir)
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "Task04_Hippocampus").exists():
            tar_path = root / "Task04_Hippocampus.tar"
            urllib.request.urlretrieve(URL, tar_path)
            with tarfile.open(tar_path) as t:
                t.extractall(root)

    def setup(self, stage=None):
        root = Path(self.hparams.data_dir) / "Task04_Hippocampus"
        imgs = sorted(p for p in (root / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))
        pairs = [(str(p), str(root / "labelsTr" / p.name)) for p in imgs]
        full = VolDataset(pairs, tuple(self.hparams.patch_size))
        n_val = int(len(full) * self.hparams.val_split)
        self.train_ds, self.val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(42))

    def _dl(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory)

    def train_dataloader(self):
        return self._dl(self.train_ds, True)

    def val_dataloader(self):
        return self._dl(self.val_ds, False)

    def test_dataloader(self):
        return self._dl(self.val_ds, False)