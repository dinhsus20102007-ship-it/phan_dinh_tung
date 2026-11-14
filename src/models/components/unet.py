import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


def build_unet(in_channels=1, num_classes=2, dim=2, n_stages=5,
               features=(32, 64, 128, 256, 320)):
    conv_op = nn.Conv2d if dim == 2 else nn.Conv3d
    norm_op = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d
    net = PlainConvUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=list(features[:n_stages]),
        conv_op=conv_op,
        kernel_sizes=[3] * n_stages,
        strides=[1] + [2] * (n_stages - 1),
        n_conv_per_stage=[2] * n_stages,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    net.apply(InitWeights_He(1e-2))
    return net