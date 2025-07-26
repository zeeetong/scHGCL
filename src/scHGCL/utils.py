import pandas as pd
import numpy as np
import torch
from model.Model import scLHGCL_encoder
import torch.nn as nn
from learnable_gumbel_hypergraph import print_hypergraph_info



def l1_distance(imputed_data, original_data):

    return np.mean(np.abs(original_data-imputed_data))

def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((original_data - imputed_data)**2))

def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def load_data(data_path):
    data_csv = pd.read_csv(data_path, index_col=0)
    return data_csv.values.T, data_csv.columns.values, data_csv.index.values

def impute_dropout(X, drop_rate=0.4, seed=123):
    np.random.seed(seed)
    X_zero = np.copy(X)
    i, j = np.nonzero(X_zero)
    ix = np.random.choice(len(i), int(drop_rate * len(i)), replace=False)
    X_zero[i[ix], j[ix]] = 0.0

    return X_zero

def feature_augmentation(X, mask_rate=0.2):
    X_aug = X.clone()
    non_zero_mask = (X_aug != 0)
    mask = torch.rand_like(X_aug) < mask_rate
    X_aug[non_zero_mask & mask] = 0
    return X_aug


def imputation(drop_data, choose_cell, device, alpha=1e-3, filter_noise=2):
    original_data = torch.FloatTensor(np.copy(drop_data)).to(device)
    dataImp = original_data.clone()

    for i in range(dataImp.shape[0]):
        nonzero_index = (dataImp[i] != 0)
        zero_index = (dataImp[i] == 0)
        y = original_data[i, nonzero_index]
        x = original_data[choose_cell[i]][:, nonzero_index].T
        xtx = torch.matmul(x.T, x)
        xtx += alpha * torch.eye(x.shape[-1], device=device)
        xty = torch.matmul(x.T, y.unsqueeze(-1))

        try:
            w = torch.linalg.solve(xtx, xty).squeeze()
        except RuntimeError:
            w = torch.linalg.lstsq(xtx, xty).solution.squeeze()

        impute_data = torch.matmul(original_data[choose_cell[i]][:, zero_index].T, w)
        impute_data[impute_data <= filter_noise] = 0
        dataImp[i, zero_index] = impute_data

    return dataImp.detach().cpu().numpy()

class ConstrastiveLoss(nn.Module):
    def __init__(self, cells_num, temperature):
        super().__init__()
        self.cells_num = cells_num
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(cells_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, cells_num):
        N = 2 * cells_num
        mask = torch.ones((N, N)).fill_diagonal_(0)
        for i in range(cells_num):
            mask[i, cells_num + i] = 0
            mask[cells_num + i, i] = 0
        return mask.bool()

    def forward(self, z_i, z_j):
        N = 2 * self.cells_num
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.cells_num)
        sim_j_i = torch.diag(sim, -self.cells_num)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(z.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss / N


def print_hypergraph_info(H):
    indices = (H > 0).nonzero(as_tuple=False)
    num_cells, num_genes = H.shape
    total_edges = indices.size(0)

    edge_connectivity = torch.zeros(num_genes).to(H.device)
    for g in range(num_genes):
        edge_connectivity[g] = (H[:, g] > 0).sum()

    node_participation = torch.zeros(num_cells).to(H.device)
    for c in range(num_cells):
        node_participation[c] = (H[c, :] > 0).sum()

    num_isolated = (node_participation == 0).sum().item()

    print(" Current hypergraph structure information:")
    print(f"  H.shape = {H.shape}")
    print(f"  Number of non-zero connections: {total_edges}")
    print(f"  Average number of cells per hyperedge: {edge_connectivity.mean():.2f}")
    print(f"  Average number of hyperedges per cell: {node_participation.mean():.2f}")
    print(f"  Number of isolated cells: {num_isolated}")


def training(X, gumbel_module, hyper_hidden=256, attn_hidden=128, epochs=200, temp=1.2, tau=0.5):
    model = scLHGCL_encoder(X.shape[1], hyper_hidden, attn_hidden).to(X.device)
    criterion = ConstrastiveLoss(X.shape[0], temp)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gumbel_module.parameters()), lr=3e-4
    )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        H1 = gumbel_module(X, tau=tau)
        H2 = gumbel_module(X, tau=tau)

        X1 = feature_augmentation(X, mask_rate=0.3)
        X2 = feature_augmentation(X, mask_rate=0.3)

        z1 = model(X1, H1)
        z2 = model(X2, H2)

        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")
            print_hypergraph_info(H1)
            print("=========================================")
            print_hypergraph_info(H2)

    return model




















