import torch
import torch.nn as nn

class HyperGraphConv(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.residual = nn.Identity()
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features, bias=False)

        nn.init.xavier_normal_(self.linear.weight)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight)

    def forward(self, H, X):
        D_v = torch.sum(H, dim=1)
        D_e = torch.sum(H, dim=0)
        edge_features = torch.matmul(H.T, X) / (D_e.unsqueeze(-1) + 1e-8)
        node_features = torch.matmul(H, edge_features) / (D_v.unsqueeze(-1) + 1e-8)
        out = self.linear(node_features)
        out = self.dropout(out)
        out = self.activation(out)

        res = self.residual(X)
        return out + res



class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.position_embed = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, x):
        x = x + self.position_embed.expand(x.size(0), -1)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        attn_output, _ = self.mha(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x.squeeze(1)



class scLHGCL_encoder(nn.Module):

    def __init__(self, input_size, hyper_hidden=256, attn_hidden=128,
                 num_heads=4, dropout=0.2):
        super().__init__()

        self.hyper_conv1 = HyperGraphConv(input_size, hyper_hidden, dropout)
        self.hyper_conv2 = HyperGraphConv(hyper_hidden, hyper_hidden, dropout)

        self.attention = SelfAttention(hyper_hidden, num_heads, dropout)

        self.proj = nn.Linear(hyper_hidden, attn_hidden)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, X, H):
        x1 = self.hyper_conv1(H, X)
        x2 = self.hyper_conv2(H, x1)
        attn_out = self.attention(x2)
        return self.final_dropout(self.proj(attn_out))




