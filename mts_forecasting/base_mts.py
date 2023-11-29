import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.distributions as dist

mse_loss = nn.MSELoss(reduction='none')


class LitModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 config: dict):
        super().__init__()
        self.model = model
        self.lr = config["lr"]
        self.beta = config["beta"]
        self.decay = config["decay"]
        self.dataset = config["data"]
        self.sample_time = config["sample_time"]
        self.max_epochs = config["epochs"]
        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def data_info(target: torch.tensor) -> (int, int):
        return target.shape[0:2]

    def training_step(self, data, batch_idx) -> float:
        if self.dataset == "engine":
            if self.current_epoch < self.max_epochs // 2:
                inputs, target = data['a']
            else:
                inputs, target = data['b']
        else:
            inputs, target = data

        int_steps, batch_size = self.data_info(target)
        pred, _, q = self.model(inputs, int_steps, self.sample_time, self.training)

        kl_div = dist.kl_divergence(q, dist.Normal(0, 1)).sum(-1).mean()
        recon_loss = mse_loss(pred, target).sum([-2, -1]).mean()

        loss = recon_loss + self.beta * kl_div

        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx) -> float:
        inputs, target = data
        int_steps, batch_size = self.data_info(target)
        pred, *_ = self.model(inputs, int_steps, self.sample_time, self.training)
        loss = mse_loss(pred, target).mean()
        self.log("val_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def test_step(self, data, batch_idx) -> float:
        inputs, target = data
        int_steps, batch_size = self.data_info(target)
        pred, *_ = self.model(inputs, int_steps, self.sample_time, self.training)
        loss = mse_loss(pred, target).mean()
        self.log("test_loss", loss, on_epoch=True, batch_size=batch_size, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay)
        return [optimizer], [scheduler]
