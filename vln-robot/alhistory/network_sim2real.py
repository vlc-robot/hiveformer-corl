from typing import Optional, Tuple, Literal, Union, List
import math
from einops.layers.torch import Rearrange
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_
from torch.distributions import Bernoulli
from transformers.activations import ACT2FN
from utils_sim2real import (
    Output,
    Rotation,
    BackboneOp,
    TransformerToken,
    PointCloudToken,
    GripperPose,
    ZMode,
)


# --------------------------------------------------------------------------------
# Define network layers
# --------------------------------------------------------------------------------
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
        attention_mask: Optional[torch.Tensor],
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
        instruction: bool,
        taskvar: bool,
        num_words: int,
        num_layers: int,
    ):
        super().__init__()
        self._num_layers = num_layers

        # The cross attention layer
        if self._num_layers == 1:  # For backward compatibility
            self.cross_attention: Optional[nn.Module] = CrossAttention(
                hidden_size, num_attention_heads, dropout_prob, ctx_dim=hidden_size
            )
            self.cross_output: Optional[nn.Module] = SelfOutput(
                hidden_size, hidden_size, dropout_prob
            )
            self.cross_layers: Optional[nn.ModuleList] = None
            # Self-att and FFN layer
            self.self_att = SelfAttention(hidden_size, num_attention_heads, dropout_prob)
            self.self_inter = SelfIntermediate(hidden_size, intermediate_size)
            self.self_output1 = SelfOutput(hidden_size, hidden_size, dropout_prob)
            self.self_output2 = SelfOutput(intermediate_size, hidden_size, dropout_prob)
        else:
            self.cross_output = None
            self.cross_attention = None
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

        self._instruction = instruction
        self._taskvar = taskvar

        if self._instruction:
            self.instr_position_embedding: Optional[nn.Module] = nn.Embedding(
                num_words, hidden_size
            )
            self.instr_type_embedding: Optional[nn.Module] = nn.Embedding(2, hidden_size)
            self.instr_position_norm: Optional[nn.Module] = nn.LayerNorm(hidden_size)
            self.instr_type_norm: Optional[nn.Module] = nn.LayerNorm(hidden_size)
            self.proj_instr_encoder: Optional[nn.Module] = nn.Linear(
                instr_size, hidden_size
            )
        else:
            self.instr_type_embedding = None
            self.instr_position_embedding = None
            self.instr_type_norm = None
            self.instr_position_norm = None

        if self._taskvar:
            self.task_embedding: Optional[nn.Module] = nn.Embedding(106, hidden_size)
            self.task_norm: Optional[nn.Module] = nn.LayerNorm(hidden_size)
            self.var_embedding: Optional[nn.Module] = nn.Embedding(106, hidden_size)
            self.var_norm: Optional[nn.Module] = nn.LayerNorm(hidden_size)
            self.taskvar_type_embedding: Optional[nn.Module] = nn.Embedding(
                3, hidden_size
            )
            self.taskvar_type_norm: Optional[nn.Module] = nn.LayerNorm(hidden_size)
        else:
            self.task_embedding = None
            self.var_embedding = None
            self.task_norm = None
            self.var_norm = None
            self.taskvar_type_embedding = None
            self.taskvar_type_norm = None

    def _add_taskvar(
        self,
        taskvar: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.task_embedding is None
            or self.task_norm is None
            or self.var_embedding is None
            or self.var_norm is None
            or self.taskvar_type_embedding is None
            or self.taskvar_type_norm is None
        ):
            raise RuntimeError()
        BT, K, _ = input_tensor.shape
        B = taskvar.shape[0]
        T = BT // B

        task_emb = self.task_embedding(taskvar[:, :, 0].long())
        task_emb = self.task_norm(task_emb)
        task_emb = einops.repeat(task_emb, "b k d -> (b t) k d", t=T)

        var_emb = self.var_embedding(taskvar[:, :, 1].long())
        var_emb = self.var_norm(var_emb)
        var_emb = einops.repeat(var_emb, "b k d -> (b t) k d", t=T)

        input_tensor = torch.cat([task_emb, var_emb, input_tensor], 1)

        type_id = 2 * torch.ones(K + 2).to(input_tensor.device).unsqueeze(0).long()
        type_id[:, 0] = 0
        type_id[:, 1] = 1
        type_emb = self.taskvar_type_embedding(type_id)
        type_emb = self.taskvar_type_norm(type_emb)
        type_emb = einops.repeat(type_emb, "1 kl d -> bt kl d", bt=BT)
        input_tensor += type_emb

        return input_tensor

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
        instruction: Optional[torch.Tensor],
        taskvar: Optional[torch.Tensor],
    ):
        B, T, K, C = x.shape
        # TODO save time and mem by using the padding mask

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

        if instruction is not None:
            ctx_tensor = self._add_instruction(instruction, ctx_tensor)
            num_words = instruction.shape[1]
            ctx_attn_mask = F.pad(ctx_attn_mask, (num_words, 0))

        if taskvar is not None:
            ctx_tensor = self._add_taskvar(taskvar, ctx_tensor)
            ctx_attn_mask = F.pad(ctx_attn_mask, (2, 0))

        # For backward compatibility
        # TODO clean up case == 1
        if self._num_layers == 1:
            if (
                self.cross_attention is None
                or self.cross_output is None
                or self.self_output1 is None
                or self.self_output2 is None
                or self.self_att is None
            ):
                raise RuntimeError()
            x = self.cross_attention(input_tensor, ctx_tensor, ctx_attn_mask)
            x = self.cross_output(x, input_tensor)

            # Self attention
            self_outputs = self.self_att(x)
            x = self.self_output1(self_outputs[0], x)

            # Fully connected
            self_inter = self.self_inter(x)
            x = self.self_output2(self_inter, x)

        else:
            if self.cross_layers is None:
                raise RuntimeError()
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
        film_layer=None,
        film_residual=False,
        residual=False,
    ):
        super().__init__()
        self._residual = residual
        self._film_residual = film_residual

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

        if film_layer is not None:
            self.film = film_layer
            self.film.num_params += out_channels

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)
        res = out.clone()

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "film"):
            out = self.film(out)

        if self._film_residual:
            res = out.clone()

        if hasattr(self, "activation"):
            out = self.activation(out)
            res = self.activation(res)

        if self._residual:
            return out, res
        else:
            return out


class InstrCond(nn.Module):
    def __init__(self, instr_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Linear(instr_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, ft: torch.Tensor, instr: torch.Tensor) -> torch.Tensor:
        H, W = ft.shape[-2:]

        emb = self.embedding(instr[:, :, 0].long())
        emb = self.norm(emb)
        emb = einops.repeat(emb, "b 1 d -> b 1 d h w", h=H, w=W)

        return ft * emb


class TaskVarCond(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(106, hidden_size)
        self.task_norm = nn.LayerNorm(hidden_size)
        self.var_embedding = nn.Embedding(106, hidden_size)
        self.var_norm = nn.LayerNorm(hidden_size)

    def forward(self, ft: torch.Tensor, taskvar: torch.Tensor) -> torch.Tensor:
        H, W = ft.shape[-2:]

        task_emb = self.task_embedding(taskvar[:, 0].long())
        task_emb = self.task_norm(task_emb)
        task_emb = einops.repeat(task_emb, "bt d -> bt d h w", h=H, w=W)

        var_emb = self.var_embedding(taskvar[:, 1].long())
        var_emb = self.var_norm(var_emb)
        var_emb = einops.repeat(var_emb, "bt d -> bt d h w", h=H, w=W)

        return ft * task_emb * var_emb


class FiLMModule(nn.Module):
    def __init__(
        self,
        max_steps: int,
        max_cams: int,
        mlp: bool,
        instruction: bool,
        instr_size: int = 768,
    ):
        super().__init__()
        self._max_steps = max_steps
        self._max_cams = max_cams
        self._mlp = mlp
        self._instruction = instruction
        self.num_params: int = 0
        self.cursor: int = 0
        self.gamma: Optional[torch.Tensor] = None
        self.beta: Optional[torch.Tensor] = None
        self._is_built: bool = False
        self._instr_size = instr_size

    def build(self, device):
        if self._mlp:
            self.fc_cams = nn.Sequential(
                nn.Embedding(self._max_cams, self.num_params),
                nn.ReLU(),
                nn.Linear(self.num_params, self.num_params * 2),
            ).to(device)
            self.fc_step = nn.Sequential(
                nn.Embedding(self._max_steps, self.num_params),
                nn.ReLU(),
                nn.Linear(self.num_params, self.num_params * 2),
            ).to(device)
            kaiming_uniform_(self.fc_cams[0].weight)
            kaiming_uniform_(self.fc_step[0].weight)
        else:
            self.fc_cams = nn.Embedding(self._max_cams, self.num_params * 2).to(device)
            self.fc_step = nn.Embedding(self._max_steps, self.num_params * 2).to(device)
            kaiming_uniform_(self.fc_cams.weight)
            kaiming_uniform_(self.fc_step.weight)
        # self.fc.bias.data.zero_()

        if self._instruction:
            self.proj_instr = nn.Linear(self._instr_size, 2 * self.num_params).to(device)
        else:
            self.proj_instr = None

        self._is_built = True

    def generate_params(
        self,
        step_ids: torch.Tensor,
        cam_ids: torch.Tensor,
        instruction: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_built:
            self.build(step_ids.device)
        self.counter = 0

        context = self.fc_cams(cam_ids)
        context += self.fc_step(step_ids)

        if instruction is not None and self.proj_instr is not None:
            Q = context.unsqueeze(1)
            K = self.proj_instr(instruction)
            V = K
            dk = K.shape[-1]
            QK = torch.bmm(Q, K.transpose(1, 2)) / dk
            context = torch.bmm(QK.softmax(-1), V).squeeze(1)

        self.gamma, self.beta = context.chunk(2, dim=1)
        return self.gamma, self.beta  # type: ignore

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        nc = feat.shape[1]
        if self.gamma is None or self.beta is None:
            raise RuntimeError("Use forward before")
        gamma = self.gamma[:, self.counter : self.counter + nc].unsqueeze(2).unsqueeze(3)
        beta = self.beta[:, self.counter : self.counter + nc].unsqueeze(2).unsqueeze(3)
        self.counter += nc
        out = (1 + gamma) * feat + beta
        return out


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


def normalise_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


# --------------------------------------------------------------------------------
# Define Network
# --------------------------------------------------------------------------------
class PlainUNet(nn.Module):
    def __init__(
        self,
        rot: Rotation,
        attn_weights: bool = False,
        depth: int = 3,
        dec_len: int = 16,
        film: bool = False,
        film_residual: bool = False,
        film_mlp: bool = False,
        instr_size: int = 768,
        max_episode_length: int = 10,
        gripper_pose: GripperPose = "none",
        backbone_op: BackboneOp = "cat",
        instruction: bool = False,
        cond: bool = False,
        taskvar_token: bool = False,
        temp_len: int = 0,
        z_mode: ZMode = "embed",
    ):
        super(PlainUNet, self).__init__()
        self._backbone_op = backbone_op
        self._cond = cond
        self._dec_len = dec_len
        self._film = film
        self._film_residual = film_residual
        self._gripper_pose = gripper_pose
        self._instr_size = instr_size
        self._instruction = instruction
        self._max_episode_length = max_episode_length
        self._rot = rot
        self._taskvar = taskvar_token
        self._temp_len = temp_len
        self._z_mode = z_mode

        self._num_cams = 4 if "attn" in self._gripper_pose else 3

        if self._film:
            self.film_gen: Optional[FiLMModule] = FiLMModule(
                self._max_episode_length,
                self._num_cams,
                film_mlp,
                instruction,
                instr_size,
            )
        else:
            self.film_gen = None

        # Input RGB + Point Cloud Preprocess (SiameseNet)
        self.rgb_preprocess = ConvLayer(
            self._num_cams,
            8,
            kernel_size=(3, 3),
            stride_size=(1, 1),
            apply_norm=False,
            film_layer=self.film_gen,
        )
        self.to_feat = ConvLayer(
            8,
            16,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            film_layer=self.film_gen,
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
                    film_layer=self.film_gen,
                    film_residual=self._film_residual,
                    residual=True,
                )
            )

        self.trans_decoder = nn.ModuleList()
        self.taskvar_cond = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=dec_len,
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
                if self._taskvar and self._cond:
                    self.taskvar_cond.extend([TaskVarCond(16)])
            elif i == depth - 1:
                out_channels = 16
                if self._z_mode == "imgdec":
                    out_channels += 3
                self.trans_decoder.extend(
                    [
                        nn.Sequential(
                            ConvLayer(
                                in_channels=16 * 2,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                stride_size=(1, 1),
                            ),
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            ),
                        )
                    ]
                )
                if self._taskvar:
                    self.taskvar_cond.extend([TaskVarCond(out_channels)])

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
                if self._taskvar:
                    self.taskvar_cond.extend([TaskVarCond(16)])

        if self._rot.mode == "mse":
            rot_size = self._rot.num_dims
        elif self._rot.mode == "ce":
            rot_size = 90 * self._rot.num_dims
        elif self._rot.mode == "none":
            rot_size = 1
        else:
            raise ValueError(f"Unexpected rotation value {self._rot.mode}")

        self.quat_decoder = nn.Sequential(
            ConvLayer(
                in_channels=dec_len * 3,
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
            *dense_layer(64, 1 + rot_size, apply_activation=False),
        )

        self.maps_to_coord = ConvLayer(
            in_channels=16,
            out_channels=1,
            kernel_size=(1, 1),
            stride_size=(1, 1),
            apply_norm=False,
            apply_activation=False,
        )

        if attn_weights:
            # number of steps -> number of cameras
            self.attn_weights: Optional[nn.Embedding] = nn.Embedding(
                self._max_episode_length, 3
            )
        else:
            self.attn_weights = None

        if self._instruction and self._temp_len > 0:
            self.proj_instr_tembed: Optional[nn.Linear] = nn.Linear(
                instr_size, self._temp_len
            )
        else:
            self.proj_instr_tembed = None

        if self._z_mode == "instr":
            self.z_proj_instr: Optional[nn.Module] = nn.Sequential(
                nn.Linear(instr_size, 3),
                nn.LayerNorm(3),
            )
            self.z_pos_instr: Optional[nn.Module] = nn.Sequential(
                nn.Embedding(self._max_episode_length, 3),
                nn.LayerNorm(3),
            )

            def zero_init(m):
                if isinstance(m, nn.Linear):
                    m.weight.data.fill_(0)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.Embedding):
                    m.weight.data.fill_(0)

            self.z_pos_instr.apply(zero_init)  # type: ignore
            self.z_proj_instr.apply(zero_init)  # type: ignore
        elif self._z_mode == "instr2":
            num_tasks = 106
            # # DEBUG
            # num_tasks = 100
            self.z_proj_instr = nn.Linear(instr_size, num_tasks)
            self.z_pos_instr = nn.Embedding(self._max_episode_length, 3 * num_tasks)
            self.z_pos_instr.weight.data.fill_(0)  # type: ignore
            self.z_proj_instr.weight.data.fill_(0)  # type: ignore
            self.z_proj_instr.bias.data.fill_(0)  # type: ignore
        else:
            self.z_proj_instr = None
            self.z_pos_instr = None

    def _preprocess_film(self, rgb_obs, pc_obs, padding_mask, t, z, instruction):
        if self.film_gen is None:
            return

        N = rgb_obs.shape[2]
        B, T = padding_mask.shape
        device = padding_mask.device
        src_key_padding_mask = einops.repeat(padding_mask, "b t -> b t n", n=N)

        step_ids = torch.arange(T, dtype=torch.long, device=device)
        step_ids = einops.repeat(step_ids, "t -> b t n", b=B, n=N)
        step_ids = step_ids[src_key_padding_mask]

        cam_ids = torch.arange(N, dtype=torch.long, device=device)
        cam_ids = einops.repeat(cam_ids, "n -> b t n", b=B, t=T)
        cam_ids = cam_ids[src_key_padding_mask]

        if instruction is not None:
            instruction = einops.repeat(instruction, "b k d -> b t n k d", t=T, n=N)
            instruction = instruction[src_key_padding_mask]

        self.film_gen.generate_params(step_ids, cam_ids, instruction)

    def compute_action(self, pred: Output) -> torch.Tensor:
        rotation = self._rot.compute_action(pred["rotation"])

        return torch.cat(
            [pred["position"], rotation, pred["gripper"]],
            dim=1,
        )

    def _preprocess_tembed(self, padding_mask, t, instruction) -> torch.Tensor:
        if self._instruction is None or self.proj_instr_tembed is None:
            return t[padding_mask]

        # applying dot product attention
        K = self.proj_instr_tembed(instruction)
        Q = t
        V = K
        dk = K.shape[-1]
        QK = torch.bmm(Q, K.transpose(1, 2)) / dk
        I = torch.bmm(QK.softmax(-1), V)
        return I[padding_mask]

    def forward(
        self,
        rgb_obs: torch.Tensor,
        pc_obs: torch.Tensor,
        padding_mask: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        instruction: Optional[torch.Tensor],
        gripper: torch.Tensor,
        taskvar: Optional[torch.Tensor],
    ) -> Output:
        self._preprocess_film(rgb_obs, pc_obs, padding_mask, t, z, instruction)

        # processing encoding feature
        N = rgb_obs.shape[2]
        rgb_obs = rgb_obs[padding_mask]
        pc_obs = pc_obs[padding_mask]

        rgb_obs_ = einops.rearrange(rgb_obs, "b n ch h w -> (b n) ch h w")

        rgb_obs_ = self.rgb_preprocess(rgb_obs_)

        x = self.to_feat(rgb_obs_)

        # encoding features
        enc_feat = []
        for l in self.feature_encoder:
            x, res = l(x)
            enc_feat.append(res)

        backbone = [x]
        if self._temp_len > 0:
            tembed = self._preprocess_tembed(padding_mask, t, instruction)
            tembed = einops.repeat(
                tembed, "bpad c -> (bpad n) c h w", n=N, h=x.shape[-2], w=x.shape[-1]
            )
            backbone.append(tembed)

        if self._backbone_op == "sum":
            x = torch.stack(backbone).sum(0)
        elif self._backbone_op == "max":
            x = torch.stack(backbone).max(0).values
        elif self._backbone_op == "cat":
            x = torch.cat(backbone, dim=1)
        else:
            raise ValueError(f"Unexpected backone op: {self._backbone_op}")

        return self.head(N, pc_obs, z, x, enc_feat, padding_mask, instruction, taskvar)

    def head(
        self,
        N: int,
        pc_obs: torch.Tensor,
        z: torch.Tensor,
        x,
        enc_feat,
        padding_mask,
        instruction: Optional[torch.Tensor],
        taskvar: Optional[torch.Tensor],
    ) -> Output:
        # decoding features for translation
        enc_feat.reverse()
        xtr = x  # mypy

        for i, l in enumerate(self.trans_decoder):
            if i == 0:
                xtr = self.trans_decoder[0](x)
            else:
                xtr = l(torch.cat([xtr, enc_feat[i]], dim=1))

            if self._taskvar and self._cond and taskvar is not None:
                T = padding_mask.shape[1]
                taskvar_btn = einops.repeat(taskvar, "b 1 d -> b t (n 1) d", t=T, n=N)
                taskvar_btn = taskvar_btn[padding_mask]
                taskvar_btn = einops.rearrange(taskvar_btn, "bt n d -> (bt n) d")
                xtr = self.taskvar_cond[i](xtr, taskvar_btn)

        if self._z_mode == "imgdec":
            xt = xtr[:, :-3]
        else:
            xt = xtr

        xt = self.maps_to_coord(xt)
        xt = einops.rearrange(xt, "(b n) ch h w -> b (n ch h w)", n=N, ch=1)
        xt = torch.softmax(xt / 0.1, dim=1)
        attn_map = einops.rearrange(
            xt, "b (n ch h w) -> b n ch h w", n=N, ch=1, h=128, w=128
        )

        if self.attn_weights is not None:
            B, T = padding_mask.shape
            device = padding_mask.device
            step_ids = torch.arange(T, dtype=torch.long, device=device)
            step_ids = einops.repeat(step_ids, "t -> b t", b=B)
            step_ids = step_ids[padding_mask]
            weights = self.attn_weights(step_ids)
            attn_map = attn_map * einops.repeat(
                weights, "b n -> b n ch h w", ch=1, h=128, w=128
            )

        pc_obs = einops.rearrange(pc_obs, "b n ch h w -> b n ch h w")
        position = einops.reduce(pc_obs * attn_map, "b n ch h w -> b ch", "sum")

        # decoding features for rotation
        x = einops.rearrange(x, "(b n) ch h w -> b (n ch) h w", n=N)
        xr = self.quat_decoder(x)
        task = None

        rotation = xr[:, :-1]
        if self._rot.mode == "mse":
            rotation = normalise_quat(rotation)
        elif self._rot.mode == "ce":
            rotation = einops.rearrange(
                rotation, "b (d r) -> b d r", d=self._rot.num_dims
            )
        elif self._rot.mode == "none":
            rotation = rotation
        else:
            raise ValueError(f"Unexpected {self._rot.mode}")

        if self._z_mode == "embed":
            z = z[padding_mask]
            position += z
        elif self._z_mode == "instr":
            if (
                instruction is None
                or self.z_proj_instr is None
                or self.z_pos_instr is None
            ):
                raise ValueError("no instr")
            g = instruction.mean(1)
            B, T = padding_mask.shape
            device = padding_mask.device
            z_proj = self.z_proj_instr(g)
            z_proj = einops.repeat(z_proj, "b d -> b t d", t=T)
            step_ids = torch.arange(T, dtype=torch.long, device=device)
            z_pos = self.z_pos_instr(step_ids.unsqueeze(0)).squeeze(0)
            z_pos = einops.repeat(z_pos, "t d -> b t d", b=B)
            z_instr = z_pos + z_proj
            z_instr = z_instr[padding_mask]
            position += z_instr
        elif self._z_mode == "instr2":
            if (
                instruction is None
                or self.z_proj_instr is None
                or self.z_pos_instr is None
            ):
                raise ValueError("no instr")
            g = instruction.mean(1)
            B, T = padding_mask.shape
            device = padding_mask.device

            task = self.z_proj_instr(g)
            num_tasks = task.shape[1]
            z_instr = task.softmax(1)
            z_instr = einops.repeat(z_instr, "b n -> b t 1 n", t=T)
            z_instr = z_instr[padding_mask]

            step_ids = torch.arange(T, dtype=torch.long, device=device)
            z_pos = self.z_pos_instr(step_ids.unsqueeze(0)).squeeze(0)
            z_pos = einops.repeat(z_pos, "t (n d) -> b t n d", b=B, n=num_tasks, d=3)
            z_pos = z_pos[padding_mask]

            z_offset = torch.bmm(z_instr, z_pos).squeeze(1)
            position += z_offset
        elif self._z_mode == "imgdec":
            offset = xtr[:, -3:]
            offset = einops.reduce(offset, "(b n) ch h w -> b ch", "sum", n=N)
            position += offset
        else:
            raise ValueError(self._z_mode)

        return {
            "position": position,
            "rotation": rotation,
            "gripper": torch.sigmoid(xr[:, -1:]),
            "attention": attn_map,
            "task": task,
        }


class TransformerUNet(PlainUNet):
    def __init__(
        self,
        tr_token: TransformerToken = "tnc",
        nhead=8,
        hidden_dim=64,
        dim_feedforward: int = 64,
        embed_only: bool = False,
        stateless: bool = False,
        mask_obs_prob: float = 0.0,
        no_residual: bool = False,
        num_words: int = 52,
        num_layers: int = 1,
        num_cams: int = 3,
        pcd_token: PointCloudToken = "none",
        **kwargs,
    ):
        super(TransformerUNet, self).__init__(**kwargs)

        self._embed_only = embed_only
        self._hidden_dim = hidden_dim
        self._mask_obs_prob = mask_obs_prob
        self._no_residual = no_residual
        self._num_layers = num_layers
        self._num_words = num_words
        self._pcd_token = pcd_token
        self._tr_token = tr_token
        self._stateless = stateless

        self.self_attention: Optional[nn.Module] = None
        self.cross_transformer: Optional[nn.Module] = None

        if self._embed_only:
            raise RuntimeError()
        elif self._tr_token == "tnc":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                batch_first=True,
            )
            self.self_attention = nn.TransformerEncoder(
                encoder_layer, num_layers=self._num_layers
            )
            self.visual_embedding = nn.Linear(64, self._hidden_dim)
            self.visual_norm = nn.LayerNorm(self._hidden_dim)
        elif self._tr_token == "tnhw":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                batch_first=True,
            )
            self.self_attention = nn.TransformerEncoder(
                encoder_layer, num_layers=self._num_layers
            )
            token_size = 16
            if self._pcd_token != "none":
                token_size += 3
            self.visual_embedding = nn.Linear(token_size, self._hidden_dim)
            self.visual_norm = nn.LayerNorm(self._hidden_dim)

        elif self._tr_token == "tnhw_cm_sa":
            token_size = 16
            if self._pcd_token != "none":
                token_size += 3
            self.visual_embedding = nn.Linear(token_size, self._hidden_dim)
            self.visual_norm = nn.LayerNorm(self._hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                batch_first=True,
            )
            self.self_attention = nn.TransformerEncoder(
                encoder_layer, num_layers=self._num_layers
            )

            self.cross_transformer = CrossTransformer(
                hidden_size=hidden_dim,
                intermediate_size=dim_feedforward,
                instr_size=self._instr_size,
                num_attention_heads=nhead,
                dropout_prob=0.1,
                instruction=self._instruction,
                taskvar=self._taskvar,
                num_words=self._num_words,
                num_layers=self._num_layers,
            )
        elif self._tr_token == "tnhw_cm":
            self.cross_transformer = CrossTransformer(
                hidden_size=hidden_dim,
                intermediate_size=dim_feedforward,
                instr_size=self._instr_size,
                num_attention_heads=nhead,
                dropout_prob=0.1,
                instruction=self._instruction,
                taskvar=self._taskvar,
                num_words=self._num_words,
                num_layers=self._num_layers,
            )
            token_size = 16
            if self._pcd_token != "none":
                token_size += 3
            self.visual_embedding = nn.Linear(token_size, self._hidden_dim)
            self.visual_norm = nn.LayerNorm(self._hidden_dim)
        else:
            raise ValueError(f"Unexpected {self._tr_token}")

        if "token" in self._gripper_pose:
            self.gripper_embedding: Optional[nn.Module] = nn.Linear(8, self._hidden_dim)
            self.type_embedding: Optional[nn.Module] = nn.Embedding(2, hidden_dim)
            self.type_norm: Optional[nn.Module] = nn.LayerNorm(self._hidden_dim)
        else:
            self.gripper_embedding = None
            self.type_embedding = None
            self.type_norm = None

        if self._instruction:
            self.instr_position_embedding: Optional[nn.Module] = nn.Embedding(
                self._num_words, hidden_dim
            )
            self.instr_type_embedding: Optional[nn.Module] = nn.Embedding(
                2, self._hidden_dim
            )
            self.instr_position_norm: Optional[nn.Module] = nn.LayerNorm(hidden_dim)
            self.instr_type_norm: Optional[nn.Module] = nn.LayerNorm(self._hidden_dim)
            self.proj_instr_encoder: Optional[nn.Module] = nn.Linear(
                self._instr_size, self._hidden_dim
            )
        else:
            self.instr_type_embedding = None
            self.instr_position_embedding = None
            self.instr_type_norm = None
            self.instr_position_norm = None
            self.proj_instr_encoder = None

        self.position_embedding = nn.Embedding(self._max_episode_length, hidden_dim)
        # # DEBUG
        # self.position_embedding = nn.Embedding(10, hidden_dim)
        self.cam_embedding = nn.Embedding(num_cams, self._hidden_dim)

        self.position_norm = nn.LayerNorm(hidden_dim)
        self.cam_norm = nn.LayerNorm(self._hidden_dim)

        if self._tr_token == "tnc":
            self.channel_embedding = nn.Embedding(16 + 3, self._hidden_dim)
            self.channel_norm = nn.LayerNorm(self._hidden_dim)
        elif self._tr_token in ("tnhw", "tnhw_cm", "tnhw_cm_sa"):
            self.pix_embedding = nn.Embedding(64, self._hidden_dim)
            self.pix_norm = nn.LayerNorm(self._hidden_dim)
        else:
            raise ValueError(f"Unexpected {self._tr_token}")

    def _add_instruction(self, instruction, embedding: torch.Tensor) -> torch.Tensor:
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
        t,
        z,
        instruction: Optional[torch.Tensor],
        gripper: torch.Tensor,
        taskvar: Optional[torch.Tensor],
    ) -> Output:
        padding_mask2 = torch.ones_like(padding_mask)  # HACK
        self._preprocess_film(rgb_obs, pc_obs, padding_mask2, t, z, instruction)

        # processing encoding feature
        N = rgb_obs.shape[2]
        T = rgb_obs.shape[1]
        B = rgb_obs.shape[0]
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

        backbone = []
        if not self._no_residual:
            x_pad = x[padding_mask]
            x_pad = einops.rearrange(x_pad, "bpad n c h w -> (bpad n) c h w")
            backbone = [x_pad]

        # Add extra channels with Point Clouds
        if self._pcd_token != "none":
            pcd = einops.rearrange(pc_obs, "b t n c h w -> (b t n) c h w")
            pcd = F.max_pool2d(pcd, (16, 16))
            pcd = einops.rearrange(pcd, "(b t n) c h w -> b t n c h w", b=B, t=T, n=N)
            x = torch.cat([x, pcd], 3)

        # Add history channels to the backbone
        if self._tr_token == "tnc":
            ce = self._encoder_tnc(x, padding_mask, instruction, gripper, taskvar)
        elif self._tr_token == "tnhw":
            ce = self._encoder_tnhw_sa(x, padding_mask, instruction, gripper, taskvar)
        elif self._tr_token == "tnhw_cm":
            ce = self._encoder_tnhw_cm(x, padding_mask, instruction, gripper, taskvar)
        elif self._tr_token == "tnhw_cm_sa":
            ce = self._encoder_tnhw_cm_sa(x, padding_mask, instruction, gripper, taskvar)
        else:
            raise ValueError(f"Unexpected token format {self._tr_token}")

        ce = ce[padding_mask]  # bpad n c h w
        ce = einops.repeat(ce, "bpad n c h w -> (bpad n) c h w")
        backbone.append(ce)

        # Add t embeddings to the backbone
        if t is not None and self._temp_len > 0:
            t = t[padding_mask]
            t = einops.repeat(
                t, "bpad c -> (bpad n) c h w", n=N, h=x.shape[-2], w=x.shape[-1]
            )
            backbone.append(t)

        if self._backbone_op == "sum":
            x = torch.stack(backbone).sum(0)
        elif self._backbone_op == "max":
            x = torch.stack(backbone).max(0).values
        elif self._backbone_op == "cat":
            x = torch.cat(backbone, dim=1)
        else:
            raise ValueError(f"Unexpected backone op: {self._backbone_op}")

        return self.head(
            N,
            pc_obs[padding_mask],
            z,
            x,
            enc_feat,
            padding_mask,
            instruction,
            taskvar,
        )

    def _encoder_tnc(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        instruction: Optional[torch.Tensor],
        gripper: torch.Tensor,
        taskvar: Optional[torch.Tensor],
    ):
        T, N, C = x.shape[1:4]
        K = N * C

        position = torch.arange(T).type_as(x).unsqueeze(0).long()
        pos_emb = self.position_embedding(position)
        pos_emb = self.position_norm(pos_emb).squeeze(0)
        pos_emb = einops.repeat(pos_emb, "t d -> b (t n c) d", b=x.shape[0], n=N, c=C)

        ch_id = torch.arange(C).type_as(x).unsqueeze(0).long()
        ch_emb = self.channel_embedding(ch_id)
        ch_emb = self.channel_norm(ch_emb).squeeze(0)
        ch_emb = einops.repeat(ch_emb, "c d -> b (t n c) d", b=x.shape[0], n=N, t=T)

        cam_id = torch.arange(N).type_as(x).unsqueeze(0).long()
        cam_emb = self.cam_embedding(cam_id)
        cam_emb = self.cam_norm(cam_emb).squeeze(0)
        cam_emb = einops.repeat(cam_emb, "n d -> b (t n c) d", b=x.shape[0], c=C, t=T)

        # encoding history
        xe = einops.rearrange(x, "b t n c h w -> b (t n c) (h w)", n=N, t=T)
        # xe = self.visual_embedding(xe)
        # xe = self.visual_norm(xe)
        xe += ch_emb + cam_emb + pos_emb

        causal_mask = get_causal_mask_by_block(T, K, self._stateless).to(xe.device)
        causal_mask = causal_mask.float()

        src_key_padding_mask = einops.repeat(padding_mask, "b t -> b (t n c)", n=N, c=C)

        if self._instruction and instruction is not None:
            xe = self._add_instruction(instruction, xe)
            num_words = instruction.shape[1]
            src_key_padding_mask = F.pad(src_key_padding_mask, (num_words, 0), value=True)

            causal_mask2 = torch.zeros((num_words + K * T, num_words + K * T)).type_as(xe)
            causal_mask2[num_words:, num_words:] = causal_mask
            causal_mask2[:num_words, num_words:] = -float("inf")
            causal_mask = causal_mask2

        if self.self_attention is None:
            raise RuntimeError()

        ce = self.self_attention(
            xe, mask=causal_mask, src_key_padding_mask=~src_key_padding_mask
        )

        if self._instruction and instruction is not None:
            num_words = instruction.shape[1]
            ce = ce[:, num_words:]

        h, w = x.shape[4:6]
        ce = einops.rearrange(ce, "b (t n c) (h w) -> b t n c h w", n=N, t=T, h=h)
        return ce

    def _encoder_tnhw_sa(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        instruction: Optional[torch.Tensor],
        gripper: torch.Tensor,
        taskvar: Optional[torch.Tensor],
    ):
        B, T, N, C, H, W = x.shape
        K = N * H * W

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
        xe = einops.rearrange(xe, "b t n h w d -> b t n h w d")

        if (
            "token" in self._gripper_pose
            and self.gripper_embedding is not None
            and self.type_embedding is not None
            and self.type_norm is not None
        ):
            grip_emb = self.gripper_embedding(gripper).unsqueeze(2)
            xe = einops.rearrange(xe, "b t n h w c -> b t (n h w) c")
            xe = torch.cat([grip_emb, xe], 2)
            type_id = torch.ones((B, T, N * H * W + 1)).to(x.device).long()
            type_id[:, :, 0] = 0
            type_emb = self.type_embedding(type_id)
            type_emb = self.type_norm(type_emb)
            xe += type_emb
            K = xe.shape[2]
            xe = einops.rearrange(xe, "b t k c -> b (t k) c")
        else:
            xe = einops.rearrange(xe, "b t n h w c -> b (t n h w) c")

        causal_mask = get_causal_mask_by_block(T, K, self._stateless).to(xe.device)
        causal_mask = causal_mask.float()

        src_key_padding_mask = einops.repeat(padding_mask, "b t -> b (t k)", k=K)

        if self._instruction and instruction is not None:
            xe = self._add_instruction(instruction, xe)
            num_words = instruction.shape[1]
            src_key_padding_mask = F.pad(src_key_padding_mask, (num_words, 0), value=True)

            causal_mask2 = torch.zeros((num_words + T * K, num_words + T * K)).type_as(xe)
            causal_mask2[num_words:, num_words:] = causal_mask
            causal_mask2[:num_words, num_words:] = -float("inf")
            causal_mask = causal_mask2

        if self.self_attention is None:
            raise RuntimeError()

        ce = self.self_attention(
            xe, mask=causal_mask, src_key_padding_mask=~src_key_padding_mask
        )

        if self._instruction and instruction is not None:
            num_words = instruction.shape[1]
            ce = ce[:, num_words:]

        ce = einops.rearrange(ce, "b (t k) c -> b t k c", t=T, k=K)

        if "token" in self._gripper_pose:
            ce = ce[:, :, 1:]

        ce = einops.rearrange(ce, "b t (n h w) c -> b t n c h w", n=N, h=H, w=W)

        return ce

    def _encoder_tnhw_cm(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        instruction: Optional[torch.Tensor],
        gripper: torch.Tensor,
        taskvar: Optional[torch.Tensor],
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

        if (
            "token" in self._gripper_pose
            and self.gripper_embedding is not None
            and self.type_embedding is not None
            and self.type_norm is not None
        ):
            grip_emb = self.gripper_embedding(gripper).unsqueeze(2)
            xe = einops.rearrange(xe, "b t n h w c -> b t (n h w) c")
            xe = torch.cat([grip_emb, xe], 2)
            type_id = torch.ones((B, T, N * H * W + 1)).to(x.device).long()
            type_id[:, :, 0] = 0
            type_emb = self.type_embedding(type_id)
            type_emb = self.type_norm(type_emb)
            xe += type_emb
        else:
            xe = einops.rearrange(xe, "b t n h w c -> b t (n h w) c")

        if self.cross_transformer is None:
            raise RuntimeError()
        ce = self.cross_transformer(xe, padding_mask, instruction, taskvar)

        # if self._instruction and instruction is not None:
        #     num_words = instruction.shape[1]
        #     ce = ce[:, num_words:]

        if "token" in self._gripper_pose:
            ce = ce[:, 1:]

        ce = einops.rearrange(ce, "(b t) (n h w) c -> b t n c h w", n=N, t=T, h=H, w=W)

        return ce

    def _encoder_tnhw_cm_sa(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        instruction: Optional[torch.Tensor],
        gripper: torch.Tensor,
        taskvar: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ce_sa = self._encoder_tnhw_sa(x, padding_mask, instruction, gripper, taskvar)
        ce_cm = self._encoder_tnhw_cm(x, padding_mask, instruction, gripper, taskvar)
        return torch.cat([ce_sa, ce_cm], 3)
