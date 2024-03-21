""" Define the sublayers in encoder/decoder layer """
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # n_head: 注意力头的数量
        # d_model: 模型的维度大小
        # d_k: 注意力头的键（key）的维度大小
        # d_v: 注意力头的值（value）的维度大小
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(
            d_model, n_head * d_k, bias=False
        )  # 将输入的维度由 d_model 映射到 n_head * d_k 或 n_head * d_v 的空间，以准备进行多头注意力操作
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # mask 表示注意力掩码，用于控制注意力的作用范围。如果没有传入掩码，则默认为 None。

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head  # 键的维度、值的维度和注意力头的数量
        sz_b, len_q, len_k, len_v = (
            q.size(0),
            q.size(1),
            k.size(1),
            v.size(1),
        )  # 输入张量 q 的批量大小、查询的长度、键的长度和值的长度。

        residual = q  #  将输入查询张量保存为 residual，以便后续进行残差连接

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        """通过线性变换层将查询、键和值映射到多个头上，并将结果进行维度重塑，
        使得每个头的维度为 (sz_b, len_q, n_head, d_k)、(sz_b, len_k, n_head, d_k) 和 (sz_b, len_v, n_head, d_v)。
        """
        ## .view(sz_b, len_q, n_head, d_k): view 是PyTorch中的一个方法，用于重新塑形张量
        # 将经过线性变换后的查询向量重新塑形为四个维度：
        # sz_b（批量大小），len_q（查询序列的长度），n_head（注意力头的数量，用于多头注意力机制），d_k（每个注意力头的维度大小）
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # 后面变成：size*head*矩阵
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 将查询、键和值张量的头维度转置，以便执行缩放点积注意力操作。这样，张量的维度变为 (sz_b, n_head, len_q, d_k)，键和值张量的维度也相应地变化
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (sz_b, n_head, len_q, d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.为mask增加一个新的维度

        # 调用 self.attention 实例的 forward 方法，传入查询、键、值张量和掩码，
        # 执行缩放点积注意力操作。得到的结果包括注意力加权的值张量 q 和注意力分布张量 attn
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        """
        transpose(1, 2): 交换了张量q的第1个维度和第2个维度的位置
        contiguous(): 确保张量在内存中是连续的.contiguous()方法会返回张量的一个连续内存块的副本（如果原始张量不是连续的）
        view方法会根据提供的维度参数来重新安排张量的形状。第三个维度用-1表示，这是一个占位符，意味着这个维度的大小将由PyTorch自动计算，以保证总的元素数量不变。
        """
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(
            d_in, d_hid
        )  # position-wise,将输入的维度由d_in映射到d_hid，实现了位置级别的前馈
        self.w_2 = nn.Linear(
            d_hid, d_in
        )  # position-wise,将隐藏层的维度由d_hid映射回原始的输入维度d_in，同样实现了位置级别的前馈
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)  # 随机地将部分神经元置零，以防止过拟合。

    def forward(self, x):  # 接收输入张量x，并对其进行处理。
        residual = x  # 输入张量x保存为residual，以便后续进行残差连接

        x = self.w_2(
            F.relu(self.w_1(x))
        )  # 先将输入张量x通过第一个线性变换层self.w_1进行线性变换，然后经过ReLU激活函数，再通过第二个线性变换层self.w_2进行线性变换，最终得到新的张量x。
        x = self.dropout(x)  # 对新得到的张量x进行dropout操作
        x += residual  # 将dropout后的张量x与之前保存的residual进行残差连接

        x = self.layer_norm(x)  #  对残差连接后的张量x进行Layer Normalization操作

        return x
