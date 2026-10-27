"""GPU-resident data pipelines (no CPU data-loader bottleneck)."""
import os

import numpy as np
import torch

CACHE = os.environ.get("SYSDL_CACHE", "/data/users/cs24s008/Datasets/sysdl_cache")
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


class CifarGPU:
    """Whole dataset on the GPU; augmentation = pad-4 random crop + horizontal flip."""

    def __init__(self, name, device, seed=0):
        z = np.load(os.path.join(CACHE, "cifar", f"{name}.npz"))
        to = lambda a: torch.from_numpy(a).to(device)
        self.xtr = to(z["xtr"]).permute(0, 3, 1, 2).contiguous()      # uint8 N,3,32,32
        self.ytr = to(z["ytr"].astype(np.int64))
        self.xte = to(z["xte"]).permute(0, 3, 1, 2).contiguous()
        self.yte = to(z["yte"].astype(np.int64))
        self.mean = torch.tensor(MEAN[name], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD[name], device=device).view(1, 3, 1, 1)
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self.classes = int(self.ytr.max()) + 1

    def norm(self, x):
        return (x.float() / 255.0 - self.mean) / self.std

    def train_epoch(self, batch):
        n = self.xtr.shape[0]
        perm = torch.randperm(n, device=self.device, generator=self.g)
        ar = torch.arange(32, device=self.device)
        for i in range(0, n - batch + 1, batch):             # drop last partial batch
            idx = perm[i:i + batch]
            x = torch.nn.functional.pad(self.xtr[idx], (4, 4, 4, 4))
            ox = torch.randint(0, 9, (batch,), device=self.device, generator=self.g)
            oy = torch.randint(0, 9, (batch,), device=self.device, generator=self.g)
            rows = (oy[:, None] + ar)[:, :, None]
            cols = (ox[:, None] + ar)[:, None, :]
            x = x[torch.arange(batch, device=self.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
            flip = torch.rand(batch, device=self.device, generator=self.g) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            yield self.norm(x), self.ytr[idx]

    def steps_per_epoch(self, batch):
        return self.xtr.shape[0] // batch

    def test_batches(self, batch=1000):
        for i in range(0, self.xte.shape[0], batch):
            yield self.norm(self.xte[i:i + batch]), self.yte[i:i + batch]
