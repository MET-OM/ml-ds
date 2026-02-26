import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        lr: float = 1e-3,
        batch_size: int = 16,
        num_workers: int = 4,
        criterion: nn.Module | None = None,
        enable_validation: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "train_dataset", "val_dataset", "test_dataset", "criterion"]
        )

        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.enable_validation = enable_validation

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.criterion = criterion if criterion is not None else nn.MSELoss()

    def _build_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if not self.enable_validation or self.val_dataset is None:
            return None
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return self._build_loader(self.test_dataset, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        monitor_metric = (
            "val_loss" if self.enable_validation and self.val_dataset is not None else "train_loss"
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                threshold_mode="abs",
                factor=0.1,
                patience=5,
                threshold=0.001,
                min_lr=1e-5,
            ),
            "monitor": monitor_metric,
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
