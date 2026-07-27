import torch
import torch.nn as nn
from .transformer import build_transformer


class A_T(nn.Module):
  def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, num_classes, **kwargs):
    super().__init__()

    self.input_dim = input_dim

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
    
    x_a = self.trans_audio_net_1(x_a, key_padding_mask = x_a_padding_mask)
    x_a = x_a.transpose(1, 2)
    x_a = self.avgpool(x_a).squeeze(dim = -1)
    
    x_t = self.trans_text_net_1(x_t, key_padding_mask = x_t_padding_mask)
    x_t = x_t.transpose(1, 2)
    x_t = self.avgpool(x_t).squeeze(dim = -1)
    
    # x = x_a * x_t
    x = torch.cat((x_a, x_t), dim = 1) # 以相加的方式融合


    embedding_FC1 = self.FC1(x)
    embedding_FC2 = self.FC2(embedding_FC1)
    embedding_FC3 = self.FC3(embedding_FC2)
    out = self.classifier(embedding_FC3) 


    return out


