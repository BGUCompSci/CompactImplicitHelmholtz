import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, n_channels: int, out_channels: int, channels_sizes: list[int], lr=0.001, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.n_channels = n_channels
        self.out_channels = out_channels
        self.learning_rate = lr

        self.inc = DoubleConvBlock(n_channels, channels_sizes[0])
        self.downs = nn.ModuleList([Down(channel, channel * 2) for channel in channels_sizes])

    def forward(self, x):
        x = self.inc(x)
        down_features = [x]

        for down in self.downs:
            x = down(x)
            down_features.append(x)

        return down_features


class ImplicitUNet(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lr=1e-3,
                 implicit: bool = True, small=False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = lr
        self.implicit = implicit

        self.downsample1 = ConvBlock(self.in_channels, 8, 5, 2, padding=2)

        self.expand1 = ConvBlock(8, 16, kernel_size=1)
        self.depth_wise1 = ConvBlock(16, 16, kernel_size=3, groups=16, padding=1)
        self.shrink1 = ConvBlock(16, 8, kernel_size=1, activation=True)

        self.downsample2 = ConvBlock(8, 16, kernel_size=5, stride=2, padding=2)

        self.expand2 = ConvBlock(16, 32, kernel_size=1)
        self.depth_wise2 = ConvBlock(32, 32, kernel_size=3, groups=32, padding=1)
        self.shrink2 = ConvBlock(32, 16, kernel_size=1, activation=True)

        self.downsample3 = ConvBlock(16, 32, kernel_size=5, stride=2, padding=2)

        self.expand3 = ConvBlock(32, 64, kernel_size=1)
        self.depth_wise3 = ConvBlock(64, 64, kernel_size=3, groups=64, padding=1)
        self.shrink3 = ConvBlock(64, 32, kernel_size=1, activation=True)

        if small:
            self.downsample4 = ConvBlock(32, 64, kernel_size=5, stride=1, padding=2)
            self.up1 = Up(64, 32, stride=1)
        else:
            self.downsample4 = ConvBlock(32, 64, kernel_size=5, stride=2, padding=2)
            self.up1 = Up(64, 32)

        if implicit:
            self.implicit_step = ImplicitStep(1e-5, 64)

        self.up2 = Up(32, 16)
        self.up3 = Up(16, 8)

        self.out_conv = OutConv(8, self.out_channels)
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        x1 = self.downsample1(x)

        x2 = self.expand1(x1)
        x3 = self.depth_wise1(x2) + encoder_features[1]
        x4 = self.shrink1(x3)

        x5 = self.downsample2(x4)

        x6 = self.expand2(x5)
        x7 = self.depth_wise2(x6) + encoder_features[2]
        x8 = self.shrink2(x7)

        x9 = self.downsample3(x8)

        x10 = self.expand3(x9)
        x11 = self.depth_wise3(x10) + encoder_features[3]
        x12 = self.shrink3(x11)

        x13 = self.downsample4(x12)

        if self.implicit:
            x13 = self.implicit_step(x13)

        x = self.up1(x13, x9)
        x = self.up2(x, x5)
        x = self.up3(x, x1)

        x = self.out_conv(x)
        x = self.up_sample(x)

        return x


class EncoderSolver(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lr=1e-3,
                 implicit: bool = True, small: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        encoder_channels = [8, 16, 32, 64, 128]

        self.encoder = Encoder(in_channels - 2, in_channels - 2, encoder_channels)
        self.solver = ImplicitUNet(in_channels - 1, out_channels, implicit=implicit, small=small)
        self.lr = lr

    def forward(self, x) -> torch.Tensor:
        kappa = (x[:, 2]).unsqueeze(1)
        encoded_features = self.encoder(kappa)
        return self.solver(x[:, :2], encoded_features)

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss = self.loss(x, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        loss = self.loss(x, y)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def loss(self, x, y):
        x_hat = self(x)
        return nn.functional.mse_loss(x_hat, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Softplus(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Softplus())

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            DoubleConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=stride)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # input is CHW
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ImplicitStep(nn.Module):
    def __init__(self, h, c_in) -> None:
        super().__init__()
        self.h = h
        real = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float)
        imag = torch.eye(3)
        self.weights = nn.Parameter(torch.stack((real, imag)).repeat(c_in // 2, 1, 1).reshape(1, c_in, 3, 3))
        self._green = None

    def forward(self, r):
        m = self.weights.shape
        n = r.shape
        width = r.shape[-1]  # b x c x H x W
        height = r.shape[-2]  # b x c x H x W
        batch_size = r.shape[0]  # b x c x H x W
        channels = r.shape[1]

        r_padded = torch.zeros(batch_size, channels, 2 * height, 2 * width, device=r.device)
        r_padded[:, :, height // 2: 3 * height // 2, width // 2: 3 * width // 2] = r[:, :]
        r_padded = torch.fft.rfft2(r_padded)

        if self.training or (self._green is None or self._green.shape[-1] != r_padded.shape[-1]):
            point_source = torch.zeros(n[-2], n[-1], dtype=torch.float, device=r.device)
            point_source[n[-2] // 2, n[-1] // 2] = 1
            mid1 = (m[-1] - 1) // 2
            mid2 = (m[-2] - 1) // 2
            Kp = torch.zeros(m[1], 3 * n[-2], 3 * n[-1], dtype=torch.float, device=r.device)
            Kp[:, :mid1 + 1, :mid2 + 1] = self.weights[:, :, mid1:, mid2:]
            Kp[:, -mid1:, :mid2 + 1] = self.weights[:, :, :mid1, -(mid2 + 1):]
            Kp[:, :mid1 + 1, -mid2:] = self.weights[:, :, -(mid1 + 1):, :mid2]
            Kp[:, -mid1:, -mid2:] = self.weights[:, :, :mid1, :mid2]
            k_hat = torch.fft.rfft2(Kp)

            b_pad = torch.zeros(3 * n[-2], 3 * n[-1], dtype=torch.float, device=r.device)
            b_pad[n[-2]:n[-2] * 2, n[-1]:n[-1] * 2] = point_source
            b_hat = torch.fft.rfft2(b_pad)

            t = k_hat / ((k_hat ** 2) + self.h)
            xKh = (b_hat * t)

            xKh = torch.fft.irfft2(xKh)[:, n[-2] // 2: 5 * n[-2] // 2, n[-1] // 2: 5 * n[-1] // 2]
            green = xKh

            green = torch.fft.fftshift(green)
            green = torch.fft.rfft2(green)

            self._green = green

        # Transform the residual to its complex form
        u = torch.fft.irfft2(self._green * r_padded)

        # Output should be b x 2 x H x W
        r = u[:, :, height // 2: 3 * height // 2, width // 2: 3 * width // 2]
        return r


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1,
                 padding: int = 0, activation=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.Softplus())
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, padding=padding),
                nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
