import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RareGraph: GNN benchmark on rare classes")

    # Task Type Arguments
    parser.add_argument('--task', type=str, default='node', choices=['node', 'edge'], help='Task Type(Node Classification or Link Prediction)')

    # Dataset Arguments
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Citeseer', 'Pubmed', 'NCI1', 'Proteins', 'Reddit', 'Wiki', 'Refinitiv'] , help='Dataset Name (Refinitiv requires licensed data that is not included; see data/Refinitiv/README.md)')

    # Imbalance options. The names --im_class_num, --class_sample_num and --up_scale follow
    # GraphSMOTE (Zhao et al., WSDM 2021); --imb_ratio follows GraphENS (Park et al., ICLR 2022).
    parser.add_argument('--imb_setting', type=str, default='upsample', choices=['upsample','smote'], help='Minority oversampling method')
    
    parser.add_argument('--imb_ratio', type=float, default=1, help='Imbalance ratio: training nodes of a minority class / training nodes of a majority class')

    parser.add_argument('--class_sample_num', type=int, default=20, help='Number of samples per class')

    parser.add_argument('--im_class_num', type=int, default=1, help='Number of classes to imbalance')

    parser.add_argument('--up_scale', type=int, default=1, help='Oversampling scale: synthetic nodes per minority training node (0 = set from the mean class size)')

    # Model Arguments
    parser.add_argument('--model', type=str, default='gcn', choices = ['sage', 'gcn', 'gat', 'gatv2'], help='Model Name')

    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN Layers')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden Dimension')

    parser.add_argument('--gat_heads', type=int, default=8, help='Number of GAT Heads')
    
    parser.add_argument('--gatv2_heads', type=int, default=8, help='Number of GATv2 Heads')

    # Training Arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device Name')

    parser.add_argument('--seed', type=int, default=42, help='Random Seed')

    parser.add_argument('--epochs', type=int, default=100, help='Number of Training Epochs')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')

    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight Decay')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')

    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'], help='Loss Function')

    args = parser.parse_args()

    return args
