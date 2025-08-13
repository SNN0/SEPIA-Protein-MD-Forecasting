
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint

# --------------------------------------------------------------------------
class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, grad_ckpt=False):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.grad_ckpt = grad_ckpt
    def forward(self, x, edge_index):
        def fn(x, ei): return F.relu(self.conv(x, ei))
        if self.grad_ckpt and self.training:
            return torch.utils.checkpoint.checkpoint(fn, x, edge_index)
        return fn(x, edge_index)

class GraphRNN(nn.Module):
    def __init__(self, node_dim, hidden, n_atoms, n_ss,
                 n_layers=4, dropout=0.2, grad_ckpt=False):
        super().__init__()
        self.gcn = nn.ModuleList([
            GCNBlock(node_dim if i==0 else hidden, hidden, grad_ckpt)
            for i in range(n_layers)
        ])
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.coord_head = nn.Linear(hidden, n_atoms*3)
        self.ss_head    = nn.Linear(hidden, n_atoms*n_ss)
        self.dropout    = nn.Dropout(dropout)
        self.n_atoms    = n_atoms
        self.n_ss       = n_ss

    def forward(self, seq_batch):
        embeds = []
        for batch in seq_batch:
            x, ei = batch.x, batch.edge_index
            for g in self.gcn:
                x = g(x, ei)
            x = global_mean_pool(x, batch.batch)
            embeds.append(x)
        seq = torch.stack(embeds, dim=1)  # (B,W,hidden)
        out, _ = self.gru(seq)
        coord = self.coord_head(out).view(out.size(0), out.size(1), self.n_atoms, 3)
        ss    = self.ss_head(out).view(out.size(0), out.size(1), self.n_atoms, self.n_ss)
        return coord, ss