import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from losses import *
from models import *
import utils
from args import parse_args
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, confusion_matrix 
from sklearn.metrics import precision_recall_curve, auc

from torch_geometric.utils import negative_sampling
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from utils import CustomDataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


# Get Arguments
args = parse_args()
seed = args.seed


# Set Device
device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
print('Running on device:', device)

dataset_name = args.dataset
path = os.path.join(os.getcwd(), 'data', dataset_name)
data = utils.load_data(dataset_name, path)
num_class = data.y.max().item() + 1
edge_index = data.edge_index
# Debugging
print('edge index', data.edge_index)
print('data', data)
print('data', data.y[:20])
class_counts = np.bincount(data.y)
print('Class counts:', class_counts)

# Class-imbalanced split and minority oversampling.
# Implements the method of GraphSMOTE (Zhao et al., WSDM 2021); independent implementation.
# The `--im_class_num` classes with the largest labels are the minority classes: each one
# gets int(class_sample_num * imb_ratio) training nodes; every other class gets class_sample_num.
n_classes = data.y.max().item() + 1
first_minority = n_classes - args.im_class_num
train_per_class = [
    args.class_sample_num if cls < first_minority else int(args.class_sample_num * args.imb_ratio)
    for cls in range(n_classes)
]
idx_train, idx_val, idx_test, class_num_mat = utils.split_arti(data.y, train_per_class)

# Oversample the minority classes in input-feature space ('upsample' or 'smote').
oversample = {'upsample': utils.src_upsample, 'smote': utils.src_smote}[args.imb_setting]
adj, features, labels, idx_train = oversample(
    data.adj, data.x, data.y, idx_train, portion=args.up_scale, im_class_num=args.im_class_num)

#if required uncomment and check
# edge_index, edge_attr = dense_to_sparse(adj)
# data = Data(x=features, edge_index = edge_index, edge_attr = edge_attr, y = labels, adj = adj, train_idx = idx_train, val_idx = idx_val, test_idx = idx_test, class_num_mat = class_num_mat )
num_nodes = len(labels)

# Initialize masks
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Set the indices to True based on your splits
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True

data = Data(x=features, y = labels, adj = adj, edge_index = edge_index, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask, class_num_mat = class_num_mat )


# Checking after imbalancing
num_class = data.y.max().item() + 1
print('edge index', data.edge_index)
print('data', data)
print('data', data.y[:20])
class_counts = np.bincount(data.y)
print('Class counts:', class_counts)
data = data.to(device)

# Task Type
task_type = args.task

# Model Selection
if task_type == 'node':
    if args.model == 'gcn':
        model = GCNNodeClassification(args.num_layers, data.num_features, args.hidden_dim, num_class).to(device)
    elif args.model == 'sage':
        model = SAGENodeClassification(args.num_layers, data.num_features, args.hidden_dim, num_class).to(device)
    elif args.model == 'gat':
        model = GATNodeClassification(args.num_layers, data.num_features, args.hidden_dim, num_class, args.gat_heads).to(device)
    elif args.model == 'gatv2':
        model = GATv2NodeClassification(args.num_layers, data.num_features, args.hidden_dim, num_class, args.gatv2_heads).to(device)
else:  # task_type == 'edge'
    if args.model == 'gcn':
        model = GCNLinkPrediction(args.num_layers, data.num_features, args.hidden_dim).to(device)
    elif args.model == 'sage':
        model = SAGELinkPrediction(args.num_layers, data.num_features, args.hidden_dim).to(device)
    elif args.model == 'gat':
        model = GATLinkPrediction(args.num_layers, data.num_features, args.hidden_dim, args.gat_heads).to(device)
    elif args.model == 'gatv2':
        model = GATv2LinkPrediction(args.num_layers, data.num_features, args.hidden_dim, args.gatv2_heads).to(device)

# Training Arguments
if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'focal':
    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
else:
    raise NotImplementedError('Loss Function not supported')
epochs = args.epochs
batch_size = args.batch_size
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
dataset = CustomDataset(data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if task_type == 'node':
    dataset = CustomDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
elif task_type == 'edge':
    pass

# Training for Node Classification
def train_node(epoch):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
    return total_loss / len(data_loader.dataset)

# Testing for Node Classification
def test_node(epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    y_true = []
    y_pred = []
    y_scores = []
    for batch in data_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        pred = out.argmax(dim=1)
        y_true.append(batch.y[batch.test_mask].cpu().numpy())
        y_pred.append(pred[batch.test_mask].cpu().numpy())
        y_scores.append(F.softmax(out[batch.test_mask], dim=1).cpu().detach().numpy())
        loss = criterion(out[batch.test_mask], batch.y[batch.test_mask])
        total_loss += loss.item() * batch.num_nodes
        total_correct += int((pred[batch.test_mask] == batch.y[batch.test_mask]).sum())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_scores = np.concatenate(y_scores)
    
    acc = (y_pred == y_true).mean()
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    macro_auc = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovr')
    
    
    
    # Handle multi-class confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    epsilon = 1e-8
    sensitivity = np.diag(cm) / (np.sum(cm, axis=1) + epsilon)
    specificity = np.diag(cm) / (np.sum(cm, axis=0) + epsilon)
    gmean = np.sqrt(sensitivity * specificity) 
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    # Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)

    return total_loss / len(data_loader.dataset), acc, balanced_acc, f1, f1_macro, f1_weighted, auc, macro_auc, gmean, kappa, mcc

# Training for Link Prediction
def train_edge(epoch):
    model.train()
    total_loss = 0
    for batch in edge_data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Perform edge splitting within the batch
        train_pos_edge_index = batch.train_pos_edge_index
        train_neg_edge_index = batch.train_neg_edge_index

        # Compute node embeddings
        z = model(batch.x, batch.edge_index, batch.edge_weight)

        # Compute edge scores for positive and negative edges
        pos_score = model.decode(z, train_pos_edge_index)
        neg_score = model.decode(z, train_neg_edge_index)

        # Compute loss
        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1)
        edge_label = torch.cat([torch.ones(train_pos_edge_index.size(1)), torch.zeros(train_neg_edge_index.size(1))]).to(device)
        edge_score = torch.cat([pos_score, neg_score])
        loss = criterion(edge_score, edge_label)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(edge_data_loader.dataset)

# Testing for Link Prediction
def test_edge(epoch):
    model.eval()
    total_loss = 0
    total_auc = 0
    for batch in edge_data_loader:
        batch = batch.to(device)

        # Perform edge splitting within the batch
        test_pos_edge_index = batch.test_pos_edge_index
        test_neg_edge_index = batch.test_neg_edge_index

        # Compute node embeddings
        z = model(batch.x, batch.edge_index, batch.edge_weight)

        # Compute edge scores for positive and negative edges
        pos_score = model.decode(z, test_pos_edge_index)
        neg_score = model.decode(z, test_neg_edge_index)

        # Compute loss and AUC
        edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=1)
        edge_label = torch.cat([torch.ones(test_pos_edge_index.size(1)), torch.zeros(test_neg_edge_index.size(1))]).to(device)
        edge_score = torch.cat([pos_score, neg_score])
        loss = criterion(edge_score, edge_label)

        edge_probs = torch.sigmoid(edge_score).detach().cpu().numpy()
        edge_label = edge_label.cpu().numpy()
        auc = roc_auc_score(edge_label, edge_probs)

        total_loss += loss.item() * batch.num_graphs
        total_auc += auc * batch.num_graphs

    return total_loss / len(edge_data_loader.dataset), total_auc / len(edge_data_loader.dataset)

if task_type == 'node':
    print("Node Classification Task")
    dataset = CustomDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        loss = train_node(epoch)
        (test_loss, test_acc, test_balanced_acc, test_f1, test_f1_macro,test_f1_weighted,
         test_auc, test_macro_auc, test_gmean, test_kappa, test_mcc) = test_node(epoch)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Balanced Accuracy: {test_balanced_acc:.4f},"
                  f"Test F1: {test_f1:.4f}, Test F1 Weighted: {test_f1_weighted:.4f}, Test F1 Macro: {test_f1_macro:.4f}, Test AUC: {test_auc:.4f}, "
                  f"Test Macro AUC: {test_macro_auc:.4f}, Test GMean: [{', '.join(f'{g:.4f}' for g in test_gmean)}], "
                  f"Test Kappa: {test_kappa:.4f}, Test MCC: {test_mcc:.4f}")

elif task_type == 'edge':
    print("Link Prediction Task")
    train_ratio = 0.8

    # Shuffle the edges randomly
    num_edges = data.edge_index.size(1)
    perm = torch.randperm(num_edges)
    data.edge_index = data.edge_index[:, perm]

    train_size = int(train_ratio * num_edges)
    data.train_edge_index = data.edge_index[:, :train_size]
    data.test_edge_index = data.edge_index[:, train_size:]

    # Create negative edges for training
    data.train_neg_edge_index = negative_sampling(
        edge_index=data.train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_edge_index.size(1)
    )

    # Create negative edges for testing
    data.test_neg_edge_index = negative_sampling(
        edge_index=data.test_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_edge_index.size(1)
    )

    # Create positive edges for training and testing
    data.train_pos_edge_index = data.train_edge_index
    data.test_pos_edge_index = data.test_edge_index

    # Create a CustomDataset instance for edge task
    edge_dataset = CustomDataset(data, include_edge_weight=True)
    edge_data_loader = DataLoader(edge_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        loss = train_edge(epoch)
        test_loss, test_auc = test_edge(epoch)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
