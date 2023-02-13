from einops.layers.torch import Rearrange, Reduce
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# Define network layers
# --------------------------------------------------------------------------------
def conv_layer(
    in_channels,
    out_channels,
    kernel_size,
    stride_size,
    apply_norm=True,
    apply_activation=True,
):
    padding_size = (
        kernel_size // 2
        if isinstance(kernel_size, int)
        else (kernel_size[0] // 2, kernel_size[1] // 2)
    )

    layer = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )
    ]
    if apply_norm:
        layer += [nn.GroupNorm(1, out_channels, affine=True)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def normalise_quat(x):
    return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)


# --------------------------------------------------------------------------------
# Define Network
# --------------------------------------------------------------------------------
class PlainUNet(nn.Module):
    def __init__(self, action_size=8, depth=3, temp_len=0):
        super(PlainUNet, self).__init__()
        # Input RGB + Point Cloud Preprocess (SiameseNet)
        self.rgb_preprocess = nn.Sequential(
            *conv_layer(3, 8, kernel_size=(3, 3), stride_size=(1, 1), apply_norm=False)
        )
        self.to_feat = nn.Sequential(
            *conv_layer(8, 16, kernel_size=(1, 1), stride_size=(1, 1), apply_norm=False)
        )

        # Encoder-Decoder Network, maps to pixel location with spatial argmax
        self.feature_encoder = nn.ModuleList()
        for i in range(depth):
            self.feature_encoder.append(
                nn.Sequential(
                    *conv_layer(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=(3, 3),
                        stride_size=(2, 2),
                    ),
                )
            )

        self.trans_decoder = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            *conv_layer(
                                in_channels=16 + temp_len,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )
            else:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            *conv_layer(
                                in_channels=16 * 2,
                                out_channels=16,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )

        self.quat_decoder = nn.Sequential(
            *conv_layer(
                in_channels=(16 + temp_len) * 3,
                out_channels=64,
                kernel_size=(3, 3),
                stride_size=(2, 2),
            ),
            *conv_layer(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride_size=(2, 2)
            ),
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c h w -> b (c h w)"),
            *dense_layer(64, 64),
            *dense_layer(64, 5, apply_activation=False),
        )

        self.maps_to_coord = nn.Sequential(
            *conv_layer(
                in_channels=16,
                out_channels=1,
                kernel_size=(1, 1),
                stride_size=(1, 1),
                apply_norm=False,
                apply_activation=False,
            )
        )

    def forward(self, rgb_obs, pc_obs, t=None, z=None):
        # processing encoding feature
        n_frames = rgb_obs.shape[1] // 3
        rgb_obs_ = einops.rearrange(rgb_obs, "b (n ch) h w -> (b n) ch h w", ch=3)
        rgb_obs_ = self.rgb_preprocess(rgb_obs_)

        x = self.to_feat(rgb_obs_)

        # encoding features
        enc_feat = []
        for l in self.feature_encoder:
            x = l(x)
            enc_feat.append(x)

        if t is not None:
            t = einops.repeat(
                t, "b c -> (b n) c h w", n=n_frames, h=x.shape[-2], w=x.shape[-1]
            )
            x = torch.cat([x, t], dim=1)

        # decoding features for translation
        enc_feat.reverse()
        for i, l in enumerate(self.trans_decoder):
            if i == 0:
                xtr = l(x)
            else:
                xtr = l(torch.cat([xtr, enc_feat[i]], dim=1))

        xt = self.maps_to_coord(xtr)
        xt = einops.rearrange(xt, "(b n) ch h w -> b (n ch h w)", n=n_frames, ch=1)
        xt = torch.softmax(xt / 0.1, dim=1)
        xt = einops.rearrange(
            xt, "b (n ch h w) -> b n ch h w", n=n_frames, ch=1, h=128, w=128
        )
        pc_obs = einops.rearrange(pc_obs, "b (n ch) h w -> b n ch h w", ch=3)
        xt_ = einops.reduce(pc_obs * xt, "b n ch h w -> b ch", "sum")

        # decoding features for rotation
        x = einops.rearrange(x, "(b n) ch h w -> b (n ch) h w", n=n_frames)
        xr = self.quat_decoder(x)

        x = torch.cat(
            [xt_ + z, normalise_quat(xr[:, :4]), torch.sigmoid(xr[:, 4].unsqueeze(-1))],
            dim=1,
        )
        return x
