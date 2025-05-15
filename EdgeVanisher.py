import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.datasets import CitationFull, Coauthor
from sklearn.metrics import f1_score
import torch.nn.functional as F
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# Load the dataset
#dataset = CitationFull(root='.', name="PubMed")
dataset = Coauthor(root='.', name="Physics")
data = dataset[0].to(device)
num_edges = data.edge_index.size(1)
data.edge_y = torch.randint(0, 2, (num_edges,), device=device)
def split_edges(data, train_ratio=0.6, val_ratio=0.2):
    num_edges = data.edge_index.shape[1]
    indices = np.random.permutation(num_edges)
    train_cutoff = int(train_ratio * num_edges)
    val_cutoff = int((train_ratio + val_ratio) * num_edges)
    data.edge_train_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    data.edge_val_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    data.edge_test_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    data.edge_train_mask[indices[:train_cutoff]] = True
    data.edge_val_mask[indices[train_cutoff:val_cutoff]] = True
    data.edge_test_mask[indices[val_cutoff:]] = True
split_edges(data)
class BaseEdgeModel(torch.nn.Module):
    def fit(self, data, epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            src, dst = data.edge_index[:, data.edge_train_mask]
            edge_out = (out[src] * out[dst]).sum(dim=1)
            edge_y = data.edge_y[data.edge_train_mask].float()
            loss = criterion(edge_out, edge_y)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                f1 = self.evaluate(data, 'val')
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val F1: {f1:.4f}')
    @torch.no_grad()
    def evaluate(self, data, split='test'):
        self.eval()
        mask = getattr(data, f'edge_{split}_mask')
        if mask.sum() == 0:
            return 0
        out = self(data.x, data.edge_index)
        src, dst = data.edge_index[:, mask]
        edge_out = (out[src] * out[dst]).sum(dim=1).sigmoid()
        preds = (edge_out > 0.5).long()
        labels = data.edge_y[mask]
        return f1_score(labels.cpu(), preds.cpu(), average='macro')
    def edge_unlearn(self, data, edges_to_remove, epochs=50):
        remaining_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
        remaining_mask[edges_to_remove] = False
        data.edge_index = data.edge_index[:, remaining_mask]
        data.edge_y = data.edge_y[remaining_mask]
        data.edge_train_mask = data.edge_train_mask[remaining_mask]
        data.edge_val_mask = data.edge_val_mask[remaining_mask]
        data.edge_test_mask = data.edge_test_mask[remaining_mask]
        start_time = time.time()
        self.fit(data, epochs)
        end_time = time.time()
        print(f'Edge unlearning took {end_time - start_time:.2f} seconds.')
class GCN(BaseEdgeModel):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
class GAT(BaseEdgeModel):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
class GraphSAGE(BaseEdgeModel):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
class GIN(BaseEdgeModel):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GINConv(torch.nn.Linear(in_dim, hidden_dim))
        self.conv2 = GINConv(torch.nn.Linear(hidden_dim, out_dim))
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
def train_and_unlearn(model, data, edges_to_remove):
    print(f"Training {model.__class__.__name__}...")
    model.fit(data, epochs=100)
    f1_before = model.evaluate(data, 'test')
    print(f'F1 score before unlearning: {f1_before:.4f}')
    print(f"Unlearning {len(edges_to_remove)} edges...")
    model.edge_unlearn(data, edges_to_remove, epochs=50)
    f1_after = model.evaluate(data, 'test')
    print(f'F1 score after unlearning: {f1_after:.4f}')
models = {
    'GCN': GCN(data.num_features, 64, 2).to(device),
    #'GAT': GAT(data.num_features, 64, 2).to(device),
    #'GraphSAGE': GraphSAGE(data.num_features, 64, 2).to(device),
    'GIN': GIN(data.num_features, 64, 2).to(device)
}
num_remove = int(0.05 * num_edges)
edges_to_remove = np.random.choice(num_edges, num_remove, replace=False)
for name, model in models.items():
    print(f"\n===== Running {name} =====")
    train_and_unlearn(model, data.clone(), edges_to_remove)
