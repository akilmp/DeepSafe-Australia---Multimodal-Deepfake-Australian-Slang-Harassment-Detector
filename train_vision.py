import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.io import read_video
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import AUROC


class VideoDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        video, _, _ = read_video(str(path), pts_unit="sec")
        frame = video[0].permute(2, 0, 1).float() / 255.0
        if self.transform:
            frame = self.transform(frame)
        return frame, label


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/video", batch_size=8):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.transform = T.Compose(
            [
                T.Resize(weights.transforms().crop_size),
                T.CenterCrop(weights.transforms().crop_size),
                T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
            ]
        )

    def _gather(self, split):
        samples = []
        for label, cls in enumerate(["real", "fake"]):
            for path in (self.data_dir / cls / split).glob("*.mp4"):
                samples.append((path, label))
        return samples

    def setup(self, stage=None):
        self.train_ds = VideoDataset(self._gather("train"), self.transform)
        self.val_ds = VideoDataset(self._gather("val"), self.transform)
        self.test_ds = VideoDataset(self._gather("test"), self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4)


class DeepFakeModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.model = efficientnet_v2_s(weights=weights)
        self.model.classifier[1] = nn.Linear(1280, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.auroc = AUROC(num_classes=2)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage="train"):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        auc = self.auroc(probs, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_auc", auc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    dm = VideoDataModule(batch_size=args.batch_size)
    model = DeepFakeModel(lr=args.lr)

    ckpt = ModelCheckpoint(
        dirpath=".", filename="best", monitor="val_auc", mode="max", save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=16,
        callbacks=[ckpt],
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
