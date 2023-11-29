import torch.nn as nn
import lightning.pytorch as pl
from latent_dynamics.utils import *

mse_loss = nn.MSELoss(reduction='none')


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 config: dict):
        super().__init__()
        self.model = model
        self.epochs = config["epochs"]
        self.learning_rate = config["lr"]
        self.sample_time = config["sample_time"]
        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> None:
        pass

    def post_process(self, batch):
        inp, gt = batch
        trg_len = gt.shape[1]
        t = torch.linspace(0, trg_len * self.sample_time, trg_len + 1, device=self.device)
        return inp, gt, t

    def training_step(self, data, batch_idx) -> float:
        inp, gt, t = self.post_process(data)
        pred = self.model(t, inp)
        loss = mse_loss(pred, gt).sum(1).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=inp.shape[0], prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx) -> float:
        inp, gt, t = self.post_process(data)
        pred = self.model(t, inp)

        mse = mse_loss(pred, gt)
        loss = mse.sum(1).mean()
        norm_loss = (loss / pred.shape[1]) / gt.square().mean()
        accuracy = pixel_accuracy(pred, gt)

        log_dict = {"val_acc": accuracy, "norm_loss": norm_loss}

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, data, batch_idx) -> float:
        inp, gt, t = self.post_process(data)
        pred = self.model(t, inp)

        loss = mse_loss(pred, gt).mean() / gt.square().mean()
        accuracy = pixel_accuracy(pred, gt)

        log_dict = {"norm_mse": loss, "test_acc": accuracy}

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                                      total_iters=self.epochs)
        return [optimizer], [scheduler]
