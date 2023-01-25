import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEncoding(nn.Module):
    """
    Relative positional encoding (ROPE inspired rotary pe).
    """
    def __init__(self, feature_dim, pe_type):
        super().__init__()

        self.feature_dim = feature_dim
        self.pe_type = pe_type

    @staticmethod
    def embed_rotary(x, cos, sin):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    def forward(self, x_position):
        raise NotImplementedError


class RotaryPositionEncoding3D(RotaryPositionEncoding):
    def __init__(self, feature_dim, pe_type='Rotary3D'):
        super().__init__(feature_dim, pe_type)

    def forward(self, XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape
        vox = XYZ
        x_position, y_position, z_position = vox[..., 0:1], vox[..., 1:2], vox[..., 2:3]
        div_term = torch.exp(torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device) * (
                    -math.log(10000.0) / (self.feature_dim // 3)))
        div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        sinx, cosx, siny, cosy, sinz, cosz = map(lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
                                                 [sinx, cosx, siny, cosy, sinz, cosz])
        sin_pos = torch.cat([sinx, siny, sinz], dim=-1)
        cos_pos = torch.cat([cosx, cosy, cosz], dim=-1)
        position_code = torch.stack([cos_pos, sin_pos], dim=-1)

        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code


class RotaryPositionEncoding3DWrapper(nn.Module):

    def __init__(self, feature_dim):
        super().__init__()
        self.pos_embed = RotaryPositionEncoding3D(feature_dim)

    def forward(self, features, XYZ):
        """
        Return positional encoding for features.

        Arguments:
            features: downscale image feature map of shape
             (batch_size, channels, height, width)
            XYZ: point cloud in world coordinates aligned to image features of
             shape (batch_size, 3, height * downscaling, width * downscaling)
        """
        h_downscaling = XYZ.shape[-2] / features.shape[-2]
        w_downscaling = XYZ.shape[-1] / features.shape[-1]
        assert h_downscaling == w_downscaling and int(h_downscaling) == h_downscaling

        print()
        print("RotaryPositionEncoding3DWrapper:")
        XYZ = F.interpolate(XYZ, scale_factor=1. / h_downscaling, mode='bilinear')
        XYZ = einops.rearrange(XYZ, "b c h w -> b (h w) c")
        pos_code = self.pos_embed(XYZ)
        print("XYZ.shape", XYZ.shape)
        print("XYZ.min(), XYZ.max()", XYZ.min(), XYZ.max())
        print("pos_code.shape", pos_code.shape)
        print("pos_code.min(), pos_code.max()", pos_code.min(), pos_code.max())
        # TODO Reshape XYZ to [B,N,3] to pass to self.pos_embed

        # DEBUG
        raise NotImplementedError

        return XYZ
