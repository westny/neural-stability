import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.nn import functional as F


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

    def training_step(self, data, batch_idx) -> float:
        inputs, target = data
        if inputs.dim() == 2:
            inputs.unsqueeze_(-1)
        batch_size, seq_len, feat_dim = inputs.shape

        ti = torch.tensor([0, self.sample_time], device=self.device)
        x0 = torch.zeros((batch_size, self.model.n_states), device=self.device)

        inputs = self.model.encoder(inputs)

        loss = 0
        scales = torch.linspace(0.1, 1, seq_len) ** 4

        for si in range(seq_len):
            ui = inputs[:, si]
            x0 = self.model.ode(ti, x0, ui)
            x0 = x0[-1]
            y_hat = self.model.decoder(x0)
            loss += F.cross_entropy(y_hat, target) * scales[si]

        accuracy = (y_hat.argmax(1) == target).float().mean()
        log_dict = {'train_loss': loss,
                    'train_accuracy': accuracy}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, data, batch_idx) -> float:
        inputs, target = data
        if inputs.dim() == 2:
            inputs.unsqueeze_(-1)
        batch_size, seq_len, feat_dim = inputs.shape
        ti = torch.tensor([0, self.sample_time], device=self.device)
        x0 = torch.zeros((batch_size, self.model.n_states), device=self.device)

        inputs = self.model.encoder(inputs)

        for si in range(seq_len):
            ui = inputs[:, si]
            x0 = self.model.ode(ti, x0, ui)
            x0 = x0[-1]
        y_hat = self.model.decoder(x0)

        loss = F.cross_entropy(y_hat, target)

        accuracy = (y_hat.argmax(1) == target).float().mean()

        metrics = {'test_loss': loss,
                   'test_accuracy': accuracy}

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.999), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1,
                                                      total_iters=self.epochs)
        return [optimizer], [scheduler]
