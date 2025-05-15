import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph, degree
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.datasets import Coauthor, CitationFull
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
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
dataset = Coauthor(root='.', name="Physics")
#dataset = CitationFull(root='.', name="PubMed")
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
        remaining_nodes = torch.tensor(
            [node for node in range(data.num_nodes) if node not in nodes_to_remove],
            device=device
        )
        subgraph_edge_index, _ = subgraph(remaining_nodes, data.edge_index, relabel_nodes=True)
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
        print(f"Localized unlearning took {time.time() - start_time:.2f} seconds.")
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
def membership_inference_attack(model, data, attack_nodes):
    model.eval()
    logits = model(data.x, data.edge_index)
    attack_features, attack_labels = collect_attack_features(model, data, attack_nodes)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(attack_features, attack_labels, test_size=0.3, random_state=42)
    attack_model = RandomForestClassifier(random_state=42)
    attack_model.fit(X_train, y_train)
    attack_accuracy = attack_model.score(X_test, y_test)
    try:
        attack_auc = roc_auc_score(y_test, attack_model.predict_proba(X_test)[:, 1])
    except ValueError:
        attack_auc = 0
        print("Warning: Single class present in y_train or y_test. Skipping AUC-ROC calculation.")
    return attack_accuracy, attack_auc
def collect_attack_features(model, data, attack_nodes):
    logits = model(data.x, data.edge_index)
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1, keepdims=True)
    attack_features = np.hstack([
        logits[attack_nodes].detach().cpu().numpy(),
        probabilities[attack_nodes],
        entropy[attack_nodes]
    ])
    attack_labels = data.train_mask[attack_nodes].cpu().numpy().astype(int)
    return attack_features, attack_labels
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out.argmax(dim=1)
    correct = preds[data.test_mask] == data.y[data.test_mask]
    accuracy = correct.sum().item() / data.test_mask.sum().item()
    print(f'Test Accuracy: {accuracy:.4f}')
gcn = GCN(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
sage = GraphSAGE(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
gin = GIN(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
gat = GAT(dim_in=data.num_node_features, dim_h=64, dim_out=dataset.num_classes).to(device)
train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.train_mask)
val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, input_nodes=data.val_mask)

print("Training GCN...")
#gcn.fit(data, epochs=200, train_loader=train_loader, val_loader=val_loader)
#gat.fit(data, epochs=200, train_loader=train_loader, val_loader=val_loader)
#sage.fit(data, epochs=200, train_loader=train_loader, val_loader=val_loader)
gin.fit(data, epochs=200, train_loader=train_loader, val_loader=val_loader)

print("\nTesting GCN before unlearning...")
#test(gcn, data)
#test(gat, data)
#test(sage, data)
test(gin, data)
nodes_to_remove = np.random.choice(data.num_nodes, int(0.05 * data.num_nodes), replace=False).tolist()
print("\nPerforming Membership Inference Attack...")
pre_accuracy, pre_auc = membership_inference_attack(gin, data, nodes_to_remove)
print(f"Pre-Unlearning Accuracy: {pre_accuracy:.4f}, AUC-ROC: {pre_auc:.4f}")
#gcn.localized_unlearn(data, nodes_to_remove)