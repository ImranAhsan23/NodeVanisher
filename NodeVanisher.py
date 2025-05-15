import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph, degree
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.datasets import CitationFull, Coauthor, LastFMAsia, Flickr, Yelp, MovieLens, MovieLens1M, MovieLens100K
from sklearn.metrics import f1_score
import torch.nn.functional as F
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
dataset = Coauthor(root='.', name="Physics")
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
    def fit(self, data, epochs, train_loader, val_loader=None, early_stopping=10):
        best_val_f1, patience_counter = 0, 0
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                out = self(batch.x, batch.edge_index)
                loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
            val_f1 = self.evaluate(val_loader) if val_loader else 0
            self.scheduler.step(val_f1)
            mean_f1 = np.mean(val_f1)
            std_f1 = np.std(val_f1)
            print(f'Epoch {epoch + 1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}')
            if val_f1 > best_val_f1:
                best_val_f1, patience_counter = val_f1, 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print("Early stopping triggered.")
                    break
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
    def localized_unlearn(self, data, nodes_to_remove, epochs=50):
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        connected_nodes_to_remove = [node for node in nodes_to_remove if deg[node] > 0]
        if not connected_nodes_to_remove:
            print("No connected nodes found for removal; skipping unlearning.")
            return
        remaining_nodes = torch.tensor(
            [node for node in range(data.num_nodes) if node not in connected_nodes_to_remove],
            device=device
        )
        subgraph_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
        if subgraph_edge_index.size(1) == 0:
            print("No valid subgraph found after removing nodes; skipping unlearning.")
            return
        sub_data = data.__class__( 
            x=data.x[remaining_nodes],
            edge_index=subgraph_edge_index,
            y=data.y[remaining_nodes]
        ).to(device)
        sub_data.train_mask = data.train_mask[remaining_nodes]
        sub_data.val_mask = data.val_mask[remaining_nodes]
        sub_data.test_mask = data.test_mask[remaining_nodes]
        localized_loader = NeighborLoader(
            sub_data,
            num_neighbors=[3, 6],
            batch_size=batch_size,
            input_nodes=sub_data.train_mask.nonzero(as_tuple=True)[0]
        )
        print("\nRetraining on localized subgraph after unlearning...")
        start_time = time.time() 
        self.fit(data=sub_data, epochs=epochs, train_loader=localized_loader)
        end_time = time.time() 
        unlearning_time = end_time - start_time  
        print(f"Localized unlearning took {unlearning_time:.2f} seconds.") 
class GCN(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1, self.gcn2 = GCNConv(dim_in, dim_h), GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gcn1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GraphSAGE(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1, self.sage2 = SAGEConv(dim_in, dim_h), SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.sage1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GIN(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gin1, self.gin2 = GINConv(torch.nn.Linear(dim_in, dim_h)), GINConv(torch.nn.Linear(dim_h, dim_out))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gin1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gin2(x, edge_index)
        return F.log_softmax(x, dim=1)
class GAT(BaseGraphModel):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gat1, self.gat2 = GATConv(dim_in, dim_h), GATConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, x, edge_index):
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gat1(x, edge_index).relu()
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.train_mask)
val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.val_mask)
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)
    correct = preds[data.test_mask] == data.y[data.test_mask]
    accuracy = correct.sum().item() / data.test_mask.sum().item()
    print(f'Test Accuracy: {accuracy:.4f}')
def train_and_unlearn(model, data, nodes_to_remove=None):
    print(f"Training {model.__class__.__name__}...")
    model.fit(data, epochs=200, train_loader=train_loader, val_loader=val_loader)
    print(f"\nTesting {model.__class__.__name__} before unlearning...")
    test(model, data)
    if nodes_to_remove:
        print(f"Localized Unlearning {len(nodes_to_remove)} nodes from {model.__class__.__name__}...")
        model.localized_unlearn(data, nodes_to_remove)
        print(f"\nTesting {model.__class__.__name__} after unlearning...")
        test(model, data)
gcn = GCN(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
sage = GraphSAGE(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
gin = GIN(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
gat = GAT(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
num_nodes_to_remove = int(0.05 * data.num_nodes)
all_nodes = np.arange(data.num_nodes)
np.random.shuffle(all_nodes)
nodes_to_remove = all_nodes[:num_nodes_to_remove].tolist()
train_and_unlearn(gcn, data, nodes_to_remove)
train_and_unlearn(sage, data, nodes_to_remove)
train_and_unlearn(gin, data, nodes_to_remove)
train_and_unlearn(gat, data, nodes_to_remove)
