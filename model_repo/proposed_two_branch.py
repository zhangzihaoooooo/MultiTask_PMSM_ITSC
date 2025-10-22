import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from copy import deepcopy

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv1d(dim, dim * 2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.o_proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, C, T = x_q.size()

        # Q: [B, C, T] → [B, nH, T, d]
        q = self.q_proj(x_q).view(B, self.num_heads, self.head_dim, T).transpose(2, 3)

        # K, V: [B, 2C, T] → [2, B, nH, T, d]
        kv = self.kv_proj(x_kv).view(B, 2, self.num_heads, self.head_dim, T)
        k, v = kv[:, 0].transpose(2, 3), kv[:, 1].transpose(2, 3)  # [B,nH,T,d]

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B,nH,T,T]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                           # [B,nH,T,d]
        out = out.transpose(2, 3).reshape(B, C, T) # [B,C,T]
        out = self.o_proj(out)
        return self.proj_drop(out)

class ConvBNactDown(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        groups=1,
        padding=None,
    ):
        super().__init__()

        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        padding = kernel_size // 2 if padding is None else padding

        layers = [
            nn.Conv1d(
                in_channel, out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm1d(out_channel),
            nn.ELU()
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.downsample(x)
        return self.block(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        return x.transpose(-1, -2)

class Add(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        return weight[0] * x[0] + weight[1] * x[1]




class BroadcastAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=1,
                 proj_drop=0.,
                 attn_drop=0.,
                 qkv_bias=True
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads."

        self.qkv_proj = nn.Conv1d(dim, num_heads * (1 + 2 * self.head_dim), kernel_size=1, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, N = x.shape
        # [B, C, N] -> [B, num_heads * (1 + 2 * head_dim), N]
        qkv = self.qkv_proj(x).view(B, self.num_heads, 1 + 2 * self.head_dim, N)

        query, key, value = torch.split(qkv, [1, self.head_dim, self.head_dim], dim=2)

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(value) * context_vector.expand_as(value)

        out = out.permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]
        out = out.contiguous().view(B, self.num_heads, N, self.head_dim)  # [B, num_heads, N, head_dim]
        out = torch.cat([out[:, i, :, :] for i in range(self.num_heads)], dim=-1)  # [B, N, num_heads * head_dim]

        out = out.permute(0, 2, 1)  # [B, num_heads * head_dim, N] (for Conv1d input)

        # output
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BA_FFN_Block(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads=1,
                 drop=0.,
                 attn_drop=0.
                 ):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.add1 = Add()
        self.attn = BroadcastAttention(dim=dim,
                                       num_heads=num_heads,
                                       attn_drop=attn_drop,
                                       proj_drop=drop)

        self.norm2 = LayerNorm(dim)
        self.add2 = Add()
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, 1, 1, bias=True),
            nn.ELU(),
            nn.Dropout(p=drop),
            nn.Conv1d(ffn_dim, dim, 1, 1, bias=True),
            nn.Dropout(p=drop)
        )

    def forward(self, x):
        x = self.add1([self.attn(self.norm1(x)), x])
        x = self.add2([self.ffn(self.norm2(x)), x])
        return x


class LFEL(nn.Module):
    def __init__(self, d_in, d_out, drop, num_heads=1):
        super(LFEL, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv1d(d_in, d_out, kernel_size=1),
            nn.BatchNorm1d(d_out),
            nn.ELU()
        )

        self.block = BA_FFN_Block(dim=d_out,
                                  ffn_dim=d_out // 4,
                                  num_heads=num_heads,
                                  drop=drop,
                                  attn_drop=drop)

    def forward(self, x):
        x = self.proj(x)
        return self.block(x)


class Net(nn.Module):
    def __init__(self, _, in_channel=2, out_channel=4,
                 drop=0.05, dim=16, num_heads=4, seq_len=96):
        super().__init__()
        self.name = os.path.basename(__file__).split('.')[0]

        def make_branch():
            return nn.Sequential(
                ConvBNactDown(in_channel=in_channel,
                           out_channel=dim,
                           kernel_size=3,
                           stride=2),
                LFEL(dim, 2 * dim, drop, num_heads),
                LFEL(2 * dim, 4 * dim, drop, num_heads),
                nn.AdaptiveAvgPool1d(1)
            )

        self.branch_time = make_branch()
        self.branch_freq = deepcopy(self.branch_time)

        # Cross Attention
        self.cross_t2f = CrossAttention(4 * dim, num_heads)
        self.cross_f2t = CrossAttention(4 * dim, num_heads)

        self.cls_head = nn.Sequential(nn.Linear(8 * dim, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, 1))
        self.reg_head = nn.Sequential(nn.Linear(8 * dim, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, 1))

    def forward(self, x):
        x_time, x_freq = x[:, :, :2048], x[:, :, 2048:]

        f_time = self.branch_time(x_time)
        f_freq = self.branch_freq(x_freq)

        f_time = self.cross_t2f(f_time, f_freq) + f_time
        f_freq = self.cross_f2t(f_freq, f_time) + f_freq

        fused = torch.cat([f_time.flatten(1),
                           f_freq.flatten(1)], dim=1)

        pred_cls = torch.sigmoid(self.cls_head(fused))
        pred_reg = self.reg_head(fused)
        return torch.cat([pred_cls, pred_reg], dim=1)





