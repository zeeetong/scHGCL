import torch
import argparse
import pandas as pd
import utils
from learnable_gumbel_hypergraph import LearnableHypergraph, print_hypergraph_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use GPU: 1 for CUDA, 0 for CPU')
    parser.add_argument('--data_path', type=str, default='data/Zeisel_full_expression.csv', help='Path to expression matrix')
    parser.add_argument('--drop_rate', type=float, default=0.55, help='Dropout rate')
    parser.add_argument('--p', type=float, default=0.85, help='Top-p selection threshold')
    parser.add_argument('--tau', type=float, default=0.7, help='Gumbel temperature')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')

    args = parser.parse_args()

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    groundTruth, cells, genes = utils.load_data(args.data_path)
    drop_data = utils.impute_dropout(groundTruth, 0.55)
    X = torch.FloatTensor(drop_data).to(device)

    gumbel_module = LearnableHypergraph(num_cells=X.shape[0], num_genes=X.shape[1], p=args.p).to(device)
    model = utils.training(X, gumbel_module, epochs=args.epochs, tau=args.tau)

    with torch.no_grad():
        H_test = gumbel_module(X, tau=args.tau, use_gumbel=True)
        # torch.save(H_test.cpu(), 'H_test.pt')
        print_hypergraph_info(H_test)
        emb = model(X, H_test.to(device))
        sim = torch.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
        choose_cell = sim.argsort(dim=-1, descending=True)[:, 1:21]

    imputed_data = utils.imputation(drop_data, choose_cell.cpu().numpy(), device)

    print('imputed data PCCs: {:.4f}'.format(utils.pearson_corr(imputed_data, groundTruth)))
    print('imputed data L1: {:.4f}'.format(utils.l1_distance(imputed_data, groundTruth)))
    print('imputed data RMSE: {:.4f}'.format(utils.RMSE(imputed_data, groundTruth)))


if __name__ == '__main__':
    main()
