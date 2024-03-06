import torch
import torch.nn.functional as F
from torch import nn


class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=5):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.elu(x, 0.2)

        return x


class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=5):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=kernel, stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.elu(x, 0.2)
        x = self.conv(x)
        x = self.bn(x)

        return x


class TSResidualBlockI(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, padding: int = 1):
        super().__init__()

        self.norm = nn.Sequential(nn.BatchNorm2d(in_channels),
                                  nn.ELU(0.2))
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2,
                               kernel, padding=padding)
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels,
                               kernel, padding=padding)

    def forward(self, input):
        x = self.norm(input)
        x = self.conv1(x)
        x = self.bn(x)
        x = F.elu(x, 0.2)
        return input + self.conv2(x)


class TFFKappa(nn.Module):
    def __init__(self, in_channels: int, kernel: int = 3, resnet_type=TSResidualBlockI):
        super().__init__()

        self.conv_down_blocks = nn.ModuleList([ConvDown(in_channels, 16),
                                               ConvDown(16, 32),
                                               ConvDown(32, 64),
                                               ConvDown(64, 128)])

        self.conv_blocks = nn.ModuleList([resnet_type(16, 16, kernel=kernel),
                                          resnet_type(16, 16, kernel=kernel),
                                          resnet_type(32, 32, kernel=kernel),
                                          resnet_type(32, 32, kernel=kernel),
                                          resnet_type(64, 64, kernel=kernel),
                                          resnet_type(64, 64, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(16, 16, kernel=kernel),
                                          resnet_type(32, 32, kernel=kernel),
                                          resnet_type(64, 64, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel)])

        self.up_blocks = nn.ModuleList([ConvUp(128, 64),
                                        ConvUp(128, 32),
                                        ConvUp(64, 16)])

    def forward(self, input):
        # n X n X 4 X bs -> (n/2) X (n/2) X 16 X bs
        x0 = self.conv_down_blocks[0](input)
        x0 = self.conv_blocks[0](x0)
        x0 = self.conv_blocks[1](x0)

        # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
        x1 = self.conv_down_blocks[1](x0)
        x1 = self.conv_blocks[2](x1)
        x1 = self.conv_blocks[3](x1)

        # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
        x2 = self.conv_down_blocks[2](x1)
        x2 = self.conv_blocks[4](x2)
        x2 = self.conv_blocks[5](x2)

        # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
        x3 = self.conv_down_blocks[3](x2)
        x3 = self.conv_blocks[6](x3)
        x3 = self.conv_blocks[7](x3)

        # (n/16) X (n/16) X 128 X bs
        up_x3 = self.conv_blocks[8](x3)
        up_x3 = self.conv_blocks[9](up_x3)
        up_x3 = self.conv_blocks[10](up_x3)

        # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
        up_x1 = self.up_blocks[0](up_x3)
        up_x1 = torch.cat([up_x1, x2], dim=1)
        up_x1 = self.conv_blocks[14](up_x1)

        # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
        up_x2 = self.up_blocks[1](up_x1)
        up_x2 = torch.cat([up_x2, x1], dim=1)
        up_x2 = self.conv_blocks[13](up_x2)

        # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
        up_x4 = self.up_blocks[2](up_x2)
        up_x4 = torch.cat([up_x4, x0], dim=1)
        up_x4 = self.conv_blocks[12](up_x4)

        return [x0, x1, x2, up_x3, up_x1, up_x2, up_x4]


class FFSDNUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, resnet_type=TSResidualBlockI):
        super().__init__()
        self.conv_down_blocks = nn.ModuleList([ConvDown(in_channels, 16),
                                               ConvDown(32, 32),
                                               ConvDown(64, 64),
                                               ConvDown(128, 128)])

        self.conv_blocks = nn.ModuleList([resnet_type(16, 16, kernel=kernel),
                                          resnet_type(32, 32, kernel=kernel),
                                          resnet_type(64, 64, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel),
                                          resnet_type(128, 128, kernel=kernel)])

        self.up_blocks = nn.ModuleList([ConvUp(256, 64),
                                        ConvUp(256, 32),
                                        ConvUp(128, 16),
                                        nn.Sequential(nn.ELU(0.2),
                                                      nn.Conv2d(64, 16, 3, padding=1)),
                                        ConvUp(16, out_channels)])

    def forward(self, input, encoder_features):
        # n X n X 4 X bs + n X n X 16 X bs-> (n/2) X (n/2) X 16 X bs
        x0 = self.conv_down_blocks[0](input)
        x0 = self.conv_blocks[0](x0)

        # (n/2) X (n/2) X 16 X bs -> (n/4) X (n/4) X 32 X bs
        x1 = torch.cat((x0, encoder_features[0]), dim=1)
        x1 = self.conv_down_blocks[1](x1)
        x1 = self.conv_blocks[1](x1)

        # (n/4) X (n/4) X 32 X bs -> (n/8) X (n/8) X 64 X bs
        x2 = torch.cat((x1, encoder_features[1]), dim=1)
        x2 = self.conv_down_blocks[2](x2)
        x2 = self.conv_blocks[2](x2)

        # (n/8) X (n/8) X 64 X bs -> (n/16) X (n/16) X 128 X bs
        x3 = torch.cat((x2, encoder_features[2]), dim=1)
        x3 = self.conv_down_blocks[3](x3)
        x3 = self.conv_blocks[3](x3)

        # (n/16) X (n/16) X 128 X bs
        up_x3 = self.conv_blocks[4](x3)
        up_x3 = self.conv_blocks[5](up_x3)
        up_x3 = self.conv_blocks[6](up_x3)

        # (n/16) X (n/16) X 128 X bs -> (n/8) X (n/8) X 128 X bs
        up_x1 = torch.cat((up_x3, encoder_features[3]), dim=1)
        up_x1 = self.up_blocks[0](up_x1)
        up_x1 = torch.cat([up_x1, x2], dim=1)

        # (n/8) X (n/8) X 128 X bs -> (n/4) X (n/4) X 64 X bs
        up_x2 = torch.cat((up_x1, encoder_features[4]), dim=1)
        up_x2 = self.up_blocks[1](up_x2)
        up_x2 = torch.cat([up_x2, x1], dim=1)

        # (n/4) X (n/4) X 128 X bs -> (n/2) X (n/2) X 32 X bs
        up_x4 = torch.cat((up_x2, encoder_features[5]), dim=1)
        up_x4 = self.up_blocks[2](up_x4)
        up_x4 = torch.cat([up_x4, x0], dim=1)

        # (n/2) X (n/2) X 32 X bs -> (n/2) X (n/2) X 16 X bs
        up_x5 = torch.cat((up_x4, encoder_features[6]), dim=1)
        up_x5 = self.up_blocks[3](up_x5)

        # (n/2) X (n/2) X 16 X bs -> n X n X 2 X bs
        return self.up_blocks[4](up_x5)


class FeaturesUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel=3) -> None:
        super().__init__()
        self.kappa_subnet = TFFKappa(in_channels - 2, kernel=kernel)
        self.solver_subnet = FFSDNUnet(in_channels, out_channels, kernel=kernel)

    def forward(self, x) -> torch.Tensor:
        kappa = (x[:, 2]).unsqueeze(1)
        encoded_features = self.kappa_subnet(kappa)
        return self.solver_subnet(x, encoded_features)
