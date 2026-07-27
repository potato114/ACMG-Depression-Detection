import torch
import torch.nn as nn
from .transformer import build_transformer


def masked_mean(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
  valid_mask = ~padding_mask  # True = valid
  valid_mask = valid_mask.unsqueeze(-1).type_as(x)  # [B,T,1]
  x = x * valid_mask
  summed = x.sum(dim=1)
  counts = valid_mask.sum(dim=1).clamp(min=1.0)
  return summed / counts


class CrossModalGate(nn.Module):
  """
  GateModel.py 的 cross-modal gate（不改动原 transformer 结构）。

  输入:
    x_a: [B,Ta,D], x_t: [B,Tt,D]
    *_padding_mask: [B,T] (True=pad)
  输出:
    gated x_a / x_t (与输入形状一致)，可选返回门控张量 g_a/g_t
  """

  def __init__(self, dim: int):
    super().__init__()
    self.audio_gate = nn.Linear(2 * dim, 1)
    self.text_gate = nn.Linear(2 * dim, 1)
    self.norm_a = nn.LayerNorm(dim)
    self.norm_t = nn.LayerNorm(dim)

  def forward(
    self,
    x_a: torch.Tensor,
    x_t: torch.Tensor,
    a_padding_mask: torch.Tensor | None,
    t_padding_mask: torch.Tensor | None,
    return_gates: bool = False,
  ):
    if a_padding_mask is None:
      a_ctx = x_a.mean(dim=1)
    else:
      a_ctx = masked_mean(x_a, a_padding_mask)
    if t_padding_mask is None:
      t_ctx = x_t.mean(dim=1)
    else:
      t_ctx = masked_mean(x_t, t_padding_mask)

    # x_a = self.norm_a(x_a)
    # t_ctx = self.norm_t(t_ctx)
    # x_t = self.norm_t(x_t)
    # a_ctx = self.norm_a(a_ctx)

    t_ctx_e = t_ctx.unsqueeze(1).expand(-1, x_a.size(1), -1)
    a_ctx_e = a_ctx.unsqueeze(1).expand(-1, x_t.size(1), -1)

    g_a = torch.sigmoid(self.audio_gate(torch.cat([x_a, t_ctx_e], dim=-1)))
    g_t = torch.sigmoid(self.text_gate(torch.cat([x_t, a_ctx_e], dim=-1)))
    if a_padding_mask is not None:
      g_a = g_a * (~a_padding_mask).unsqueeze(-1).type_as(g_a)
    if t_padding_mask is not None:
      g_t = g_t * (~t_padding_mask).unsqueeze(-1).type_as(g_t)

    return x_a * g_a, x_t * g_t, g_a, g_t


class GateModel_A_T(nn.Module):
  def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, num_classes, **kwargs):
    super().__init__()

    self.input_dim = input_dim
    self.cross_gate = CrossModalGate(input_dim)

    self.trans_audio_net_1 = build_transformer(self_attn = True, num_layers = num_layers, embed_dim = input_dim, ffn_embed_dim = ffn_embed_dim, num_heads = num_heads)
    
    self.trans_text_net_1 = build_transformer(self_attn = True, num_layers = num_layers, embed_dim = input_dim, ffn_embed_dim = ffn_embed_dim, num_heads = num_heads)
    
    self.avgpool = nn.AdaptiveAvgPool1d(1)

    fc_dim = self.input_dim  # 1024
    
    # 1024 --> 1024
    self.FC1 = nn.Sequential(
      nn.Linear(2 * input_dim, fc_dim),
      nn.ReLU(True),
      nn.Dropout(0.5),
    )

    # 1024 --> 1024
    self.FC2 = nn.Sequential(
      nn.Linear(fc_dim, fc_dim),
      nn.ReLU(True),
      nn.Dropout(0.5),
    )

    self.FC3 = nn.Sequential(
      nn.Linear(fc_dim, fc_dim),
      nn.ReLU(True),
      nn.Dropout(0.5),
    )

    # 1024 --> num_classes
    self.classifier = nn.Sequential(
      nn.Linear(fc_dim, num_classes),
    )

  def forward(self, x_a: torch.Tensor, x_t: torch.Tensor, x_a_padding_mask, x_t_padding_mask):
    # Cross-modal gating (same gate as GateModel.py), transformer remains unchanged.
    x_a, x_t ,ga,gt= self.cross_gate(x_a, x_t, x_a_padding_mask, x_t_padding_mask)

    x_a = self.trans_audio_net_1(x_a, key_padding_mask = x_a_padding_mask)
    x_a = x_a.transpose(1, 2)
    x_a = self.avgpool(x_a).squeeze(dim = -1)
    
    x_t = self.trans_text_net_1(x_t, key_padding_mask = x_t_padding_mask)
    x_t = x_t.transpose(1, 2)
    x_t = self.avgpool(x_t).squeeze(dim = -1)
    
    # x = x_a cat x_t
    x = torch.cat((x_a, x_t), dim = 1) # 以相加的方式融合


    embedding_FC1 = self.FC1(x)
    embedding_FC2 = self.FC2(embedding_FC1)
    embedding_FC3 = self.FC3(embedding_FC2)
    out = self.classifier(embedding_FC3) 


    return out
