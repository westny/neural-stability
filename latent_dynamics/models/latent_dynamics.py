import torch.nn as nn
from typing import Union, Tuple
from models.neuralode import NeuralODE


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 128,
            n_blocks: int = 2,
            kernel_size: Union[int, Tuple[int, int, int]] = 3,
            stride: Union[int, Tuple[int, int, int]] = 1,
            padding: Union[int, Tuple[int, int, int]] = 1,
            dropout: float = 0.0
    ):
        super().__init__()
        self.depth = n_blocks
        self.net = self.block(in_channels, out_channels, n_blocks, kernel_size, stride, padding, dropout)

    @staticmethod
    def block(in_channels, out_channels, n_blocks, kernel_size, stride, padding, dropout):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                  nn.BatchNorm3d(out_channels),
                  nn.SiLU()]
        for _ in range(n_blocks - 1):
            layers += [nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
                       nn.BatchNorm3d(out_channels),
                       nn.SiLU(),
                       nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 128,
            out_channels: int = 128,
            n_blocks: int = 2,
            kernel_size: Union[int, Tuple[int, int, int]] = 3,
            stride: Union[int, Tuple[int, int, int]] = 1,
            padding: Union[int, Tuple[int, int, int]] = 1
    ):
        super().__init__()
        self.depth = n_blocks
        self.net = nn.ModuleList([self.block(in_channels, out_channels, kernel_size, stride, padding)
                                  for _ in range(n_blocks)])
        self.o = nn.SiLU()

    @staticmethod
    def block(in_channels, out_channels, kernel_size, stride, padding):
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                  nn.BatchNorm3d(out_channels),
                  nn.SiLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x_ = x
        for i in range(self.depth):
            x_ = self.net[i](x_)
            if x.shape == x_.shape:
                x_ += x
                x_ = self.o(x_)
            else:
                x = x_
        return x_


class ConvEmbed(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = x1 + x2
        return out / 1.414


class DownNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels, out_channels, dropout=dropout),
            # ResBlock(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class UpNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(),
            ConvBlock(out_channels, out_channels, kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), n_blocks=4, dropout=dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class LatentDynamics(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        latent_dim = config["n_latent"]
        dp = config["dropout"]

        self.o_act = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool3d((1, 4, 4))

        self.conv_embed = ConvEmbed(3, latent_dim // 2)
        self.down1 = DownNet(latent_dim // 2, latent_dim // 2, dropout=dp)
        self.down2 = DownNet(latent_dim // 2, latent_dim, dropout=dp)
        self.to_vec = nn.AvgPool3d(2)

        self.encoder = nn.Sequential(
            self.conv_embed,
            self.down1,
            self.down2,
            self.to_vec
        )

        self.ode = NeuralODE(n_states=latent_dim,
                             n_inputs=0,
                             n_hidden=config["n_hidden"],
                             n_layers=config["n_layers"],
                             solver=config["ode_solver"],
                             h=config["step_size"],
                             stability_init=config["stability_init"],
                             activation=config["activation"])

        self.up0 = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, latent_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(8, latent_dim),
            nn.ReLU()
        )

        self.up1 = UpNet(latent_dim, latent_dim // 2, dropout=dp)
        self.up2 = UpNet(latent_dim // 2, latent_dim // 2, dropout=dp)

        self.decoder = nn.Sequential(
            self.up0,
            self.up1,
            self.up2,
            nn.ConvTranspose3d(latent_dim // 2, 3, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
        )

    def forward(self, t, x):
        # x.shape = (N, T_in, 3, 32, 32)

        x = x.transpose(1, 2)  # x.shape = (N, 3, T_in, 32, 32)

        latent = self.encoder(x)  # latent.shape = (N, latent_dim, T_res, 4, 4)

        # Use adaptive pooling to get a fixed size latent vector
        latent = self.pool(latent).squeeze()  # latent.shape = (N, latent_dim, 1, 4, 4)

        z0 = latent.permute(0, 2, 3, 1)  # x0.shape = (N, 4, 4, latent_dim)

        zt = self.ode(t, z0)[1:]  # zt.shape = (T_out, N, 4, 4, latent_dim)

        zt = zt.permute(1, 4, 0, 2, 3)  # latent_hat.shape = (N, latent_dim, T_out, 4, 4)

        out = self.decoder(zt)  # out.shape = (N, 3, T_out, 32, 32)

        out = self.o_act(out.transpose(1, 2))  # out.shape = (N, T_out, 3, 32, 32)

        return out
