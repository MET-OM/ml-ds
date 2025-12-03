import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        lr: float = 1e-3,
        batch_size: int = 16,
        num_workers: int = 4,
        criterion=None
    ):
        """
        Generic LightningModule for training any PyTorch model.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        train_dataset : torch.utils.data.Dataset
            Dataset used for training.
        val_dataset : torch.utils.data.Dataset, optional
            Dataset used for validation (default is None).
        test_dataset : torch.utils.data.Dataset, optional
            Dataset used for testing (default is None).
        lr : float, optional
            Learning rate for the optimizer (default is 1e-3).
        batch_size : int, optional
            Batch size for all dataloaders (default is 16).
        num_workers : int, optional
            Number of workers for DataLoader (default is 4).
        criterion : torch.nn.modules.loss._Loss, optional
            Loss function to optimize (default is nn.MSELoss()).
        """
        
        super().__init__()
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.criterion = criterion if criterion is not None else nn.MSELoss()

    # ----------------------------
    # Data loaders
    # ----------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    def predict_dataloader(self):
        return self.test_dataloader()

    # ----------------------------
    # Training / Validation / Test steps
    # ----------------------------
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

    # ----------------------------
    # Optimizer
    # ----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
