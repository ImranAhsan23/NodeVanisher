import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.datasets import CitationFull, Coauthor, Flickr, LastFMAsia
from sklearn.metrics import f1_score
import torch.nn.functional as F
import time
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Load the dataset
dataset = CitationFull(root='.', name="PubMed")
data = dataset[0].to(device)
learning_rate = 0.005
dropout_rate = 0.5
num_neighbors = [5, 10]
batch_percentage = 0.2
total_nodes = data.num_nodes
batch_size = max(1, int(total_nodes * batch_percentage))
def split_data(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_cutoff = int(train_ratio * num_nodes)
    val_cutoff = int((train_ratio + val_ratio) * num_nodes)
    train_idx, val_idx, test_idx = indices[:train_cutoff], indices[train_cutoff:val_cutoff], indices[val_cutoff:]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
split_data(data)
class BaseGraphModel(torch.nn.Module):
    def fit(self, data, epochs, train_loader, val_loader=None):
        best_val_f1 = 0
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            val_f1 = self.evaluate(val_loader) if val_loader else 0
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Val F1: {val_f1:.4f}")
    @torch.no_grad()
    def evaluate(self, loader):
        self.eval()
        all_preds, all_labels = [], []
        for batch in loader:
            batch = batch.to(device)
            out = self(batch.x, batch.edge_index)
            pred_y = out.argmax(dim=1)
            all_preds.append(pred_y[batch.val_mask])
            all_labels.append(batch.y[batch.val_mask])
        if all_preds and all_labels:
            return f1_score(torch.cat(all_labels).cpu(), torch.cat(all_preds).cpu(), average='macro')
        return 0
    def retrain_from_start(self, data, epochs=200):
        print("\nRetraining from scratch after node deletion...")
        start_time = time.time()
        train_loader = NeighborLoader(data, num_neighbors=[5, 10], batch_size=batch_size, input_nodes=data.train_mask)
        val_loader = NeighborLoader(data, num_neighbors=[5, 10], batch_size=batch_size, input_nodes=data.val_mask)
        self.fit(data, epochs, train_loader, val_loader)
        retrain_f1 = self.evaluate(val_loader)
        end_time = time.time()
        retrain_time = end_time - start_time
        print(f"Retraining F1: {retrain_f1:.4f} | Time: {retrain_time:.2f} seconds")
        return retrain_f1, retrain_time
class GCN(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1, self.gcn2 = GCNConv(dim_in, dim_h), GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GAT(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gat1, self.gat2 = GATConv(dim_in, dim_h), GATConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GraphSAGE(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.conv1, self.conv2 = SAGEConv(dim_in, dim_h), SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GIN(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.conv1, self.conv2 = GINConv(torch.nn.Linear(dim_in, dim_h)), GINConv(torch.nn.Linear(dim_h, dim_out))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
models = {
    'GCN': GCN(data.num_node_features, 64, dataset.num_classes),
    'GAT': GAT(data.num_node_features, 64, dataset.num_classes),
    'GraphSAGE': GraphSAGE(data.num_node_features, 64, dataset.num_classes),
    'GIN': GIN(data.num_node_features, 64, dataset.num_classes)
}
num_nodes_to_remove = int(0.05 * data.num_nodes)
nodes_to_remove = torch.randperm(data.num_nodes)[:num_nodes_to_remove]
def retrain_after_node_deletion(model, data, nodes_to_remove):
    print(f"\n--- {model.__class__.__name__} ---")
    remaining_nodes = torch.tensor([node for node in range(data.num_nodes) if node not in nodes_to_remove], device=device)
    subgraph_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
    sub_data = data.__class__(x=data.x[remaining_nodes], edge_index=subgraph_edge_index, y=data.y[remaining_nodes]).to(device)
    sub_data.train_mask = data.train_mask[remaining_nodes]
    sub_data.val_mask = data.val_mask[remaining_nodes]
    sub_data.test_mask = data.test_mask[remaining_nodes]
    return model.retrain_from_start(sub_data)
for model_name, model in models.items():
    model.to(device)
    retrain_after_node_deletion(model, data, nodes_to_remove)
