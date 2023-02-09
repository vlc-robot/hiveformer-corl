from typing import Tuple, Literal, Union, List, Optional
import math
from einops.layers.torch import Rearrange
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, normal
from torch.distributions import Bernoulli
from transformers.activations import ACT2FN
from utils_without_rlbench import Output

# from .load_mask2former import load_mask2former
from .utils import sample_ghost_points_randomly
from .position_prediction import PositionPrediction, RelativePositionPrediction


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def generate_mask_obs(
    p: float, shape: Union[Tuple[int, ...], torch.Size]
) -> torch.Tensor:
    mask = Bernoulli(torch.ones(shape) * p).sample()
    mask[:, 0] = 0.0  # we need at least one step older
    return mask


def get_causal_mask_by_block(T: int, N: int, stateless: bool = False) -> torch.Tensor:
    """
    T: num of blocks
    N: size of a block
    """
    if stateless:
        causal_mask = torch.ones((T, T)) - torch.diag(torch.ones(T))
    else:
        causal_mask = torch.ones((T, T)).triu(diagonal=1)

    causal_mask[causal_mask == 1] = -float("inf")
    causal_mask = einops.repeat(
        causal_mask,
        "t tp -> (t n) (tp np)",
        n=N,
        np=N,
        t=T,
        tp=T,
    )
    return causal_mask


class CrossLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        dropout_prob: float,
        ctx_dim=None,
    ):
        super().__init__()
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            dropout_prob,
            ctx_dim=ctx_dim,
        )
        self.cross_output = SelfOutput(hidden_size, hidden_size, dropout_prob)
        # Self-att and FFN layer
        self.self_att = SelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.self_att_output = SelfOutput(hidden_size, hidden_size, dropout_prob)
        self.self_inter = SelfIntermediate(hidden_size, intermediate_size)
        self.self_inter_output = SelfOutput(intermediate_size, hidden_size, dropout_prob)

    def forward(
        self,
        input_tensor: torch.Tensor,
        ctx_tensor: torch.Tensor,
        ctx_attn_mask: torch.Tensor,
    ):
        x = self.cross_attention(input_tensor, ctx_tensor, ctx_attn_mask)
        x = self.cross_output(x, input_tensor)

        # Self attention
        self_outputs = self.self_att(x)
        x = self.self_att_output(self_outputs[0], x)

        # Fully connected
        self_inter = self.self_inter(x)
        x = self.self_inter_output(self_inter, x)

        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        ctx_dim=None,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in Model forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float,
        output_attentions: bool = False,
    ):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in Model forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (
            (context_layer, attention_scores)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_prob: float):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.LayerNorm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfIntermediate(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str = "gelu"
    ):
        super(SelfIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CrossTransformer(nn.Module):
    """
    Cross history layer to integrate a current observation wrt. previous time step
    Code is adapted from LXMERT.

    Cross History is composed of the following stages:
    - cross modal attention
    - self attention
    - fully connected
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        instr_size: int,
        num_attention_heads: int,
        dropout_prob: float,
        num_words: int,
        num_layers: int,
    ):
        super().__init__()
        self._num_layers = num_layers

        # The cross attention layer
        self.cross_layers = nn.ModuleList(
            [
                CrossLayer(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    dropout_prob,
                    ctx_dim=hidden_size,
                )
                for _ in range(self._num_layers)
            ]
        )

        self.instr_position_embedding = nn.Embedding(num_words, hidden_size)
        self.instr_type_embedding = nn.Embedding(2, hidden_size)
        self.instr_position_norm = nn.LayerNorm(hidden_size)
        self.instr_type_norm = nn.LayerNorm(hidden_size)
        self.proj_instr_encoder = nn.Linear(instr_size, hidden_size)

    def _add_instruction(
        self,
        instruction,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.proj_instr_encoder is None
            or self.instr_position_embedding is None
            or self.instr_position_norm is None
            or self.instr_type_embedding is None
            or self.instr_type_norm is None
        ):
            raise RuntimeError()
        instruction = self.proj_instr_encoder(instruction)
        B, num_words, D = instruction.shape
        BT, K, _ = input_tensor.shape
        T = BT // B

        position = torch.arange(num_words).type_as(input_tensor).unsqueeze(0).long()
        pos_emb = self.instr_position_embedding(position)
        pos_emb = self.instr_position_norm(pos_emb)
        pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=B)

        instruction += pos_emb
        instruction = einops.repeat(instruction, "b k d -> b t k d", t=T)
        instruction = einops.rearrange(instruction, "b t k d -> (b t) k d", t=T)

        input_tensor = torch.cat([instruction, input_tensor], 1)

        type_id = torch.ones(K + num_words).type_as(input_tensor).unsqueeze(0).long()
        type_id[:, :num_words] = 0
        type_emb = self.instr_type_embedding(type_id)
        type_emb = self.instr_type_norm(type_emb)
        type_emb = einops.repeat(type_emb, "1 kl d -> bt kl d", bt=B * T)
        input_tensor += type_emb

        return input_tensor

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        instruction: torch.Tensor,
    ):
        B, T, K, C = x.shape

        # Cross attention
        input_tensor = einops.rearrange(x, "b t k c -> (b t) k c")
        ctx_tensor = einops.rearrange(x, "b t k c -> (b t) k c")
        ctx_tensor = einops.repeat(ctx_tensor, "(b t) k c -> (b t) (tp k) c", tp=T, t=T)

        ctx_attn_mask = torch.triu(torch.ones((T, T)), diagonal=1).to(x.device)
        ctx_attn_mask = einops.repeat(ctx_attn_mask, "t tp -> (b t) tp", b=B)
        ctx_attn_mask = ctx_attn_mask.to(x.dtype)
        ctx_attn_mask[ctx_attn_mask == 1] = -float("inf")
        ctx_attn_mask = einops.repeat(
            ctx_attn_mask, "bt t -> bt nh k (t kp)", nh=1, k=K, kp=K
        )

        ctx_tensor = self._add_instruction(instruction, ctx_tensor)
        num_words = instruction.shape[1]
        ctx_attn_mask = F.pad(ctx_attn_mask, (num_words, 0))

        x = input_tensor
        for x_layer in self.cross_layers:
            x = x_layer(x, ctx_tensor, ctx_attn_mask)

        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        apply_norm=True,
        apply_activation=True,
        residual=False,
    ):
        super().__init__()
        self._residual = residual

        padding_size = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )

        if apply_norm:
            self.norm = nn.GroupNorm(1, out_channels, affine=True)

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)
        res = out.clone()

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "activation"):
            out = self.activation(out)
            res = self.activation(res)

        if self._residual:
            return out, res
        else:
            return out


def dense_layer(in_channels: int, out_channels: int, apply_activation=True):
    layer: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


class Baseline(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dim_feedforward: int = 64,
        mask_obs_prob: float = 0.0,
        num_words: int = 84,  # 42 in original code-base causes exception
        num_layers: int = 1,
        num_cams: int = 3,
        num_heads: int = 8,
        num_tasks: int = 106,
        depth: int = 4,
        instr_size: int = 512,
        max_episode_length: int = 10,
        token_size: int = 19,
        gripper_loc_bounds=None,
        sample_ghost_points=True,
        use_ground_truth_position_for_sampling=True,
        position_loss="mse",
        embedding_dim=128,
        num_ghost_point_cross_attn_layers=4,
        num_query_cross_attn_layers=4,
        relative_attention=False,
        num_ghost_points=1000,
    ):
        super(Baseline, self).__init__()

        # self.mask2former = load_mask2former()
        self.gripper_loc_bounds = gripper_loc_bounds
        self.sample_ghost_points = sample_ghost_points
        self.use_ground_truth_position_for_sampling = use_ground_truth_position_for_sampling
        assert position_loss in ["mse", "ce", "bce"]
        self.position_loss = position_loss
        self.num_ghost_points = num_ghost_points
        if relative_attention:
            self.position_prediction = RelativePositionPrediction(
                loss=position_loss,
                embedding_dim=embedding_dim,
                num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
                num_query_cross_attn_layers=num_query_cross_attn_layers,
            )
        else:
            self.position_prediction = PositionPrediction(
                loss=position_loss,
                embedding_dim=embedding_dim,
                num_ghost_point_cross_attn_layers=num_ghost_point_cross_attn_layers,
                num_query_cross_attn_layers=num_query_cross_attn_layers,
            )

        self._instr_size = instr_size
        self._max_episode_length = max_episode_length
        self._num_cams = num_cams + 1  # for Attn
        self._num_layers = num_layers
        self._num_words = num_words
        self._hidden_dim = hidden_dim
        self._mask_obs_prob = mask_obs_prob
        self._token_size = token_size

        self.cross_transformer = CrossTransformer(
            hidden_size=hidden_dim,
            intermediate_size=dim_feedforward,
            instr_size=self._instr_size,
            num_attention_heads=num_heads,
            dropout_prob=0.1,
            num_words=self._num_words,
            num_layers=self._num_layers,
        )

        self.visual_embedding = nn.Linear(token_size, self._hidden_dim)
        self.visual_norm = nn.LayerNorm(self._hidden_dim)

        self.instr_position_embedding = nn.Embedding(self._num_words, hidden_dim)
        self.instr_type_embedding = nn.Embedding(2, self._hidden_dim)
        self.instr_position_norm = nn.LayerNorm(hidden_dim)
        self.instr_type_norm = nn.LayerNorm(self._hidden_dim)
        self.proj_instr_encoder = nn.Linear(self._instr_size, self._hidden_dim)

        self.position_embedding = nn.Embedding(self._max_episode_length, hidden_dim)
        self.cam_embedding = nn.Embedding(num_cams, self._hidden_dim)

        self.position_norm = nn.LayerNorm(hidden_dim)
        self.cam_norm = nn.LayerNorm(self._hidden_dim)

        self.pix_embedding = nn.Embedding(64, self._hidden_dim)
        self.pix_norm = nn.LayerNorm(self._hidden_dim)

        # Input RGB + Point Cloud Preprocess (SiameseNet)
        self.rgb_preprocess = ConvLayer(
            self._num_cams,
            8,
            kernel_size=(3, 3),
            stride_size=(1, 1),
            apply_norm=False,
        )
        self.to_feat = ConvLayer(
            8,
            16,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
        )

        # Encoder-Decoder Network, maps to pixel location with spatial argmax
        self.feature_encoder = nn.ModuleList()
        for i in range(depth):
            self.feature_encoder.append(
                ConvLayer(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    stride_size=(2, 2),
                    residual=True,
                )
            )

        self.trans_decoder = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=80,
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
            elif i == depth - 1:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
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

            else:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
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
            ConvLayer(
                in_channels=80 * 3,
                out_channels=64,
                kernel_size=(3, 3),
                stride_size=(2, 2),
            ),
            ConvLayer(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride_size=(2, 2)
            ),
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c h w -> b (c h w)"),
            *dense_layer(64, 64),
            *dense_layer(64, 1 + 4, apply_activation=False),
        )

        self.maps_to_coord = ConvLayer(
            in_channels=16,
            out_channels=1,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            apply_activation=False,
        )

        self.z_proj_instr = nn.Linear(instr_size, num_tasks)
        self.z_pos_instr = nn.Embedding(self._max_episode_length, 3 * num_tasks)
        self.z_pos_instr.weight.data.fill_(0)  # type: ignore
        self.z_proj_instr.weight.data.fill_(0)  # type: ignore
        self.z_proj_instr.bias.data.fill_(0)  # type: ignore

    def compute_action(self, pred: Output) -> torch.Tensor:
        rotation = norm_tensor(pred["rotation"])

        return torch.cat(
            [pred["position"], rotation, pred["gripper"]],
            dim=1,
        )

    def _add_instruction(self, instruction, embedding: torch.Tensor) -> torch.Tensor:
        instruction = self.proj_instr_encoder(instruction)
        B, num_words, D = instruction.shape
        K = embedding.shape[1]

        position = torch.arange(num_words, dtype=torch.long)
        position = position.unsqueeze(0).to(embedding.device)
        pos_emb = self.instr_position_embedding(position)
        pos_emb = self.instr_position_norm(pos_emb)
        pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=B)

        instruction += pos_emb

        embedding = torch.cat([instruction, embedding], 1)

        type_id = torch.ones(K + num_words).type_as(embedding).unsqueeze(0).long()
        type_id[:, :num_words] = 0
        type_emb = self.instr_type_embedding(type_id)
        type_emb = self.instr_type_norm(type_emb)
        type_emb = einops.repeat(type_emb, "1 kl d -> b kl d", b=B)

        embedding += type_emb

        return embedding

    def forward(
        self,
        rgb_obs,
        pc_obs,
        padding_mask,
        instruction: torch.Tensor,
        gripper: torch.Tensor,
        gt_action: Optional[torch.Tensor] = None
    ) -> Output:
        padding_mask2 = torch.ones_like(padding_mask)  # HACK

        # processing encoding feature
        B, T, N = rgb_obs.shape[:3]
        device = rgb_obs.device

        rgb_obs_ = einops.rearrange(rgb_obs, "b t n ch h w -> (b t n) ch h w")

        rgb_obs_ = self.rgb_preprocess(rgb_obs_)

        x = self.to_feat(rgb_obs_)

        # encoding features
        enc_feat = []
        for l in self.feature_encoder:
            x, res = l(x)

            res = einops.rearrange(res, "(b t n) c h w -> b t n c h w", n=N, t=T)
            res = res[padding_mask]
            res = einops.rearrange(res, "bpad n c h w -> (bpad n) c h w")
            enc_feat.append(res)

        x = einops.rearrange(x, "(b t n) c h w -> b t n c h w", n=N, t=T)

        # random masking
        mask_obs = generate_mask_obs(self._mask_obs_prob, (B, T)).to(device)
        x[mask_obs.bool()] = 0

        x_pad = x[padding_mask]
        x_pad = einops.rearrange(x_pad, "bpad n c h w -> (bpad n) c h w")
        backbone = [x_pad]

        # Add extra channels with Point Clouds
        pcd = einops.rearrange(pc_obs, "b t n c h w -> (b t n) c h w")
        pcd = F.avg_pool2d(pcd, 16)
        pcd = einops.rearrange(pcd, "(b t n) c h w -> b t n c h w", b=B, t=T, n=N)
        x = torch.cat([x, pcd], 3)

        # Add history channels to the backbone
        ce = self.encoding(x, padding_mask, instruction, gripper)
        ce = ce[padding_mask]  # bpad n c h w
        ce = einops.repeat(ce, "bpad n c h w -> (bpad n) c h w")
        backbone.append(ce)

        x = torch.cat(backbone, dim=1)

        return self.head(
            N,
            pc_obs,
            rgb_obs,
            x,
            enc_feat,
            padding_mask,
            instruction,
            gripper,
            gt_action
        )

    def encoding(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        instruction: torch.Tensor,
        gripper: torch.Tensor,
    ):
        B, T, N, C, H, W = x.shape

        position = torch.arange(T).type_as(x).unsqueeze(0).long()
        pos_emb = self.position_embedding(position)
        pos_emb = self.position_norm(pos_emb).squeeze(0)
        pos_emb = einops.repeat(pos_emb, "t d -> b t n h w d", b=B, n=N, h=H, w=W)

        pix_id = torch.arange(H * W).type_as(x).unsqueeze(0).long()
        pix_emb = self.pix_embedding(pix_id)
        pix_emb = self.pix_norm(pix_emb).squeeze(0)
        pix_emb = einops.repeat(
            pix_emb, "(h w) d -> b t n h w d", b=B, n=N, t=T, h=H, w=W
        )

        cam_id = torch.arange(N).type_as(x).unsqueeze(0).long()
        cam_emb = self.cam_embedding(cam_id)
        cam_emb = self.cam_norm(cam_emb).squeeze(0)
        cam_emb = einops.repeat(cam_emb, "n d -> b t n h w d", b=B, h=H, w=W, t=T)

        # encoding history
        xe = einops.rearrange(x, "b t n c h w -> b t n h w c")
        xe = self.visual_embedding(xe)
        xe = self.visual_norm(xe)
        xe += pix_emb + cam_emb + pos_emb

        xe = einops.rearrange(xe, "b t n h w c -> b t (n h w) c")

        ce = self.cross_transformer(xe, padding_mask, instruction)

        ce = einops.rearrange(ce, "(b t) (n h w) c -> b t n c h w", n=N, t=T, h=H, w=W)

        return ce

    # def head(
    #     self,
    #     N: int,
    #     pc_obs: torch.Tensor,
    #     rgb_obs: torch.Tensor,
    #     x,
    #     enc_feat,
    #     padding_mask,
    #     instruction: torch.Tensor,
    #     gripper: torch.Tensor,
    #     gt_action
    # ) -> Output:
    #
    #     # print()
    #     # print()
    #     # print("INPUTS:")
    #     # print("pc_obs", pc_obs.shape)
    #     # print("rgb_obs", rgb_obs.shape)
    #     # print("enc_feat - contains features at different layers of a down-sampling UNet encoding RGB")
    #     # print("(batch x history x cameras) x channels x height x width")
    #     # for i in range(len(enc_feat)):
    #     #     print("enc_feat[i]", enc_feat[i].shape)
    #     # print("x - max downsampling, with point cloud info, contextualized with instruction and history")
    #     # print("x", x.shape)
    #     # print()
    #
    #     pc_obs = pc_obs[padding_mask]
    #
    #     # Position contact
    #     # enc_feat.reverse()
    #     # xtr = x
    #     # print("ORIGINAL POSITION CONTACT:")
    #     # print("heatmap (batch x history x cameras) x channels x height x width -", xtr.shape)
    #     # for i, l in enumerate(self.trans_decoder):
    #     #     if i == 0:
    #     #         xtr = self.trans_decoder[0](x)
    #     #     else:
    #     #         xtr = l(torch.cat([xtr, enc_feat[i]], dim=1))
    #     #     print("heatmap", xtr.shape)
    #     # xt = xtr
    #     # xt = self.maps_to_coord(xt)
    #     # print("heatmap", xt.shape)
    #     # xt = einops.rearrange(xt, "(b n) ch h w -> b (n ch h w)", n=N, ch=1)
    #     # print("heatmap", xt.shape)
    #     # print("heatmap.min(), heatmap.max()", xt.min(), xt.max())
    #     # xt = torch.softmax(xt / 0.1, dim=1)
    #     # print("heatmap.min(), heatmap.max()", xt.min(), xt.max())
    #     # attn_map = einops.rearrange(
    #     #     xt, "b (n ch h w) -> b n ch h w", n=N, ch=1, h=128, w=128
    #     # )
    #     # print("heatmap", attn_map.shape)
    #     # position_contact = einops.reduce(pc_obs * attn_map, "b n ch h w -> b ch", "sum")
    #     # print("position_contact", position_contact.shape)
    #     # print()
    #
    #     # Position with Mask2Former - concatenate camera images side by side
    #     # and sample ghost points
    #     # print()
    #     # print()
    #     # print("MASK2FORMER POSITION")
    #     imgs = einops.rearrange(rgb_obs, "b t n d h w -> (b t) n d h w")
    #     imgs = (imgs / 2 + 0.5) * 255.0  # Rescale to [0, 255]
    #     imgs = imgs[:, :, :3, :, :]
    #     pcds = pc_obs
    #
    #     # Sample ghost points
    #     if self.sample_ghost_points:
    #
    #         if self.use_ground_truth_position_for_sampling and gt_action is not None:
    #             # Training time
    #
    #             # Sample ghost points evenly across the workspace
    #             grid_pcd = sample_ghost_points(self.gripper_loc_bounds)
    #             grid_pcd = torch.from_numpy(grid_pcd).float().to(pcds.device)
    #             bs, num_points = pcds.shape[0], grid_pcd.shape[0]
    #             grid_pcd = grid_pcd.unsqueeze(0).repeat(bs, 1, 1)
    #
    #             # Sample the ground-truth position as an additional ghost point
    #             ground_truth_pcd = einops.rearrange(gt_action, "b t c -> (b t) c")[:, :3].unsqueeze(1).detach()
    #
    #             ghost_points_pcds = torch.cat([grid_pcd, ground_truth_pcd], dim=1)
    #
    #         else:
    #             # Inference time
    #
    #             # Sample ghost points evenly across the workspace
    #             grid_pcd = sample_ghost_points(self.gripper_loc_bounds)
    #             grid_pcd = torch.from_numpy(grid_pcd).float().to(pcds.device)
    #             bs, num_points = pcds.shape[0], grid_pcd.shape[0]
    #             grid_pcd = grid_pcd.unsqueeze(0).repeat(bs, 1, 1)
    #             ghost_points_pcds = grid_pcd
    #
    #     else:
    #         ghost_points_pcds = None
    #
    #     proprioception = einops.rearrange(gripper, "b n c -> (b n) c")[:, :3]
    #
    #     # print("pcds", pcds.shape)
    #     # print("ghost_points_pcds", ghost_points_pcds.shape)
    #     (
    #         img_attn_map,
    #         ghost_points_attn_map,
    #         intermediate_img_attn_maps,
    #         intermediate_ghost_points_attn_maps
    #     ) = self.mask2former(
    #         imgs, pcds=pcds, ghost_points_pcds=ghost_points_pcds, proprioception=proprioception
    #     )
    #     # print("img_attn_map", img_attn_map.shape)
    #     # print("ghost_points_attn_map", ghost_points_attn_map.shape)
    #
    #     attn_map = einops.rearrange(img_attn_map, "bt d h nw -> bt d (h nw)")
    #     if self.sample_ghost_points:
    #         attn_map = torch.cat([attn_map, ghost_points_attn_map], dim=-1)
    #     attn_map_pre_softmax = attn_map
    #
    #     # Compute intermediate attn maps to apply loss at every layer of Transformer
    #     # decoder - doing this quick and dirty, we'll clean up later
    #     intermediate_attn_maps_pre_softmax = []
    #     for i in range(len(intermediate_img_attn_maps)):
    #         m = einops.rearrange(intermediate_img_attn_maps[i], "bt d h nw -> bt d (h nw)")
    #         if self.sample_ghost_points:
    #             m = torch.cat([m, intermediate_ghost_points_attn_maps[i]], dim=-1)
    #         intermediate_attn_maps_pre_softmax.append(m)
    #
    #     attn_map = torch.softmax(attn_map, dim=-1)
    #     # print("attn_map", attn_map.shape)
    #
    #     all_pcds = einops.rearrange(pcds, "bt n d h w -> bt d (h n w)")
    #     if self.sample_ghost_points:
    #         ghost_points_pcds = einops.rearrange(ghost_points_pcds, "bt num_points d -> bt d num_points")
    #         all_pcds = torch.cat([all_pcds, ghost_points_pcds], dim=-1)
    #     # print("all_pcds", all_pcds.shape)
    #
    #     # Compute top points for visualization (only last batch idx = latest timestep
    #     # at inference time)
    #     # TODO Improve selection of points used to visualize attention
    #     top_attn_idxs = attn_map.topk(k=500, dim=-1).indices[-1, 0]
    #     top_points = all_pcds[-1, :, top_attn_idxs].transpose(1, 0)
    #
    #     if self.position_loss == "mse":
    #         # Take weighted sum of all points
    #         position = einops.reduce(attn_map * all_pcds, "bt d N -> bt d", "sum")
    #     elif self.position_loss in ["ce", "bce"]:
    #         # Select top point
    #         indices = attn_map.max(dim=-1).indices.squeeze(-1)
    #         position = all_pcds[torch.arange(len(indices)), :, indices]
    #     # print("position", position.shape)
    #
    #     g = instruction.mean(1)
    #     task = self.z_proj_instr(g)
    #
    #     # Position offset
    #     if not self.sample_ghost_points:
    #         B, T = padding_mask.shape
    #         device = padding_mask.device
    #         num_tasks = task.shape[1]
    #         z_instr = task.softmax(1)
    #         z_instr = einops.repeat(z_instr, "b n -> b t 1 n", t=T)
    #         z_instr = z_instr[padding_mask]
    #         step_ids = torch.arange(T, dtype=torch.long, device=device)
    #         z_pos = self.z_pos_instr(step_ids.unsqueeze(0)).squeeze(0)
    #         z_pos = einops.repeat(z_pos, "t (n d) -> b t n d", b=B, n=num_tasks, d=3)
    #         z_pos = z_pos[padding_mask]
    #         z_offset = torch.bmm(z_instr, z_pos).squeeze(1)
    #         position += z_offset
    #
    #     # Rotation
    #     x = einops.rearrange(x, "(b n) ch h w -> b (n ch) h w", n=N)
    #     xr = self.quat_decoder(x)
    #     rotation = xr[:, :-1]
    #     rotation = normalise_quat(rotation)
    #
    #     # DEBUG
    #     # raise NotImplementedError
    #
    #     return {
    #         "position": position,
    #         "rotation": rotation,
    #         "gripper": torch.sigmoid(xr[:, -1:]),
    #         "attention": attn_map_pre_softmax,
    #         "intermediate_attention": intermediate_attn_maps_pre_softmax,
    #         "points": all_pcds.detach(),
    #         "task": task,
    #         "top_points": top_points
    #     }

    def head(
        self,
        N: int,
        pc_obs: torch.Tensor,
        rgb_obs: torch.Tensor,
        x,
        enc_feat,
        padding_mask,
        instruction: torch.Tensor,
        gripper: torch.Tensor,
        gt_action
    ) -> Output:

        visible_pcd = pc_obs[padding_mask]

        visible_rgb = einops.rearrange(rgb_obs, "b t n d h w -> (b t) n d h w")
        visible_rgb = (visible_rgb / 2 + 0.5) * 255  # From [-1, 1] to [0, 255]
        visible_rgb = visible_rgb[:, :, :3, :, :]

        if self.use_ground_truth_position_for_sampling and gt_action is not None:
            # Training time

            # Sample ghost points randomly across the workspace
            grid_pcd = sample_ghost_points_randomly(self.gripper_loc_bounds, num_points=self.num_ghost_points)
            grid_pcd = torch.from_numpy(grid_pcd).float().to(visible_pcd.device)
            bs, num_points = visible_pcd.shape[0], grid_pcd.shape[0]
            grid_pcd = grid_pcd.unsqueeze(0).repeat(bs, 1, 1)

            # Sample the ground-truth position as an additional ghost point
            ground_truth_pcd = einops.rearrange(gt_action, "b t c -> (b t) c")[:, :3].unsqueeze(1).detach()

            ghost_pcd = torch.cat([grid_pcd, ground_truth_pcd], dim=1)

        else:
            # Inference time

            # Sample ghost points randomly across the workspace
            grid_pcd = sample_ghost_points_randomly(self.gripper_loc_bounds)
            grid_pcd = torch.from_numpy(grid_pcd).float().to(visible_pcd.device)
            bs, num_points = visible_pcd.shape[0], grid_pcd.shape[0]
            grid_pcd = grid_pcd.unsqueeze(0).repeat(bs, 1, 1)
            ghost_pcd = grid_pcd

        curr_gripper_position = einops.rearrange(gripper, "b t c -> (b t) c")[:, :3]

        # print()
        # print("visible_rgb", visible_rgb.shape, visible_rgb.min(), visible_rgb.max())
        # print("visible_pcd", visible_pcd.shape, visible_pcd.min(), visible_pcd.max())
        # print("curr_gripper_pos", curr_gripper_position.shape, curr_gripper_position.min(), curr_gripper_position.max())
        # print("ghost_pcd", ghost_pcd.shape, ghost_pcd.min(), ghost_pcd.max())

        position_pred = self.position_prediction(
            visible_rgb=visible_rgb,
            visible_pcd=visible_pcd,
            curr_gripper=curr_gripper_position,
            ghost_pcd=ghost_pcd
        )

        # print()
        # print("pred['position']", position_pred["position"].shape, position_pred["position"].min(),
        #       position_pred["position"].max())
        # print("position_pred['visible_rgb_masks'][-1]", position_pred['visible_rgb_masks'][-1].shape,
        #       position_pred['visible_rgb_masks'][-1].min(), position_pred['visible_rgb_masks'][-1].max())
        # print("position_pred['ghost_pcd_masks'][-1]", position_pred['ghost_pcd_masks'][-1].shape,
        #       position_pred['ghost_pcd_masks'][-1].min(), position_pred['ghost_pcd_masks'][-1].max())
        # print("position_pred['all_masks'][-1]", position_pred['all_masks'][-1].shape,
        #       position_pred['all_masks'][-1].min(), position_pred['all_masks'][-1].max())

        g = instruction.mean(1)
        task = self.z_proj_instr(g)

        # Rotation
        x = einops.rearrange(x, "(b n) ch h w -> b (n ch) h w", n=N)
        xr = self.quat_decoder(x)
        rotation = xr[:, :-1]
        rotation = normalise_quat(rotation)

        return {
            "visible_rgb_masks": position_pred["visible_rgb_masks"],
            "ghost_pcd_masks": position_pred["ghost_pcd_masks"],
            "all_masks": position_pred["all_masks"],
            "all_pcd": position_pred["all_pcd"],
            "position": position_pred["position"],
            "rotation": rotation,
            "gripper": torch.sigmoid(xr[:, -1:]),
            "task": task,
        }
