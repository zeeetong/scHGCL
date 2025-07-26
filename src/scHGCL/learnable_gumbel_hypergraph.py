# learnable_gumbel_hypergraph.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableHypergraph(nn.Module):
    def __init__(self, num_cells, num_genes, hidden_dim=128, p=0.9):
        super().__init__()
        self.linear1 = nn.Linear(1, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.p = p

    def forward(self, expression_matrix, tau=1.0, use_gumbel=True):
        """
        Args:
            expression_matrix: torch.Tensor (num_cells, num_genes), float
            tau: temperature for Softmax
            use_gumbel: whether to use Gumbel noise (True for training, False for inference)
        Returns:
            H: torch.Tensor (num_cells, num_genes), binary dense incidence matrix
        """
        num_cells, num_genes = expression_matrix.shape
        H = torch.zeros_like(expression_matrix)

        for g in range(num_genes):
            expr = expression_matrix[:, g].unsqueeze(1)  # (num_cells, 1)
            h = F.relu(self.linear1(expr))
            logits = self.linear2(h).squeeze(1)  # (num_cells,)

            if use_gumbel:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                scores = F.softmax((logits + gumbel_noise) / tau, dim=0)
            else:
                scores = F.softmax(logits / tau, dim=0)

            # Top-p selection
            sorted_y, indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(sorted_y, dim=0)
            mask = (cumulative_probs <= self.p)
            if not torch.any(mask):
                mask[0] = True

            selected_indices = indices[mask]

            hard_y = torch.zeros_like(scores)
            hard_y[selected_indices] = 1.0
            H[:, g] = hard_y

        return H


def print_hypergraph_info(H):
    indices = (H > 0).nonzero(as_tuple=False)
    num_cells, num_genes = H.shape
    total_edges = indices.size(0)

    edge_connectivity = torch.sum(H > 0, dim=0).float()
    node_participation = torch.sum(H > 0, dim=1).float()
    num_isolated = (node_participation == 0).sum().item()

    print(" 当前超图结构信息:")
    print(f"  H.shape = {H.shape}")
    print(f"  非零连接数: {total_edges}")
    print(f"  每个超边连接的平均细胞数: {edge_connectivity.mean():.2f}")
    print(f"  每个细胞参与的超边数: {node_participation.mean():.2f}")
    print(f"  孤立细胞数: {num_isolated}")
