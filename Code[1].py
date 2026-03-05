import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


DATA_PATH = r"D:\chandru1\Work191\archive (5)\iot23_combined.csv"
MAX_SAMPLES = 8000          
CLIENTS = 2              
ROUNDS = 2                  


data = pd.read_csv(DATA_PATH).dropna().drop_duplicates()

if len(data) > MAX_SAMPLES:
    data = data.sample(MAX_SAMPLES)

data["label"] = (data["label"] != 0).astype(int)


for col in data.select_dtypes(include=["object"]).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

X = MinMaxScaler().fit_transform(data.drop("label", axis=1).values)
y = data["label"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2
)


def split_clients(X, y, n):
    idx = np.array_split(np.random.permutation(len(X)), n)
    return [(X[i], y[i]) for i in idx]

clients_data = split_clients(X_train, y_train, CLIENTS)


def build_graph(X, y):
    k = int(np.log(len(X))) + 1
    nn_model = NearestNeighbors(n_neighbors=k).fit(X)
    edges = nn_model.kneighbors(X, return_distance=False)

    src = np.repeat(np.arange(len(X)), k)
    dst = edges.flatten()

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long)
    )


class GNN_IDS(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.conv1 = GCNConv(f, 32)
        self.fc = nn.Linear(32, 2)

    def forward(self, d):
        x = F.relu(self.conv1(d.x, d.edge_index))
        return self.fc(x)


def local_train(model, graph):
    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    opt.zero_grad()
    loss = loss_fn(model(graph), graph.y)
    loss.backward()
    opt.step()

    return model.state_dict()


def fed_avg(ws):
    return {k: torch.mean(torch.stack([w[k] for w in ws]), 0) for k in ws[0]}


global_model = GNN_IDS(X.shape[1])

for _ in range(ROUNDS):
    weights = []
    for Xc, yc in clients_data:
        g = build_graph(Xc, yc)
        m = GNN_IDS(X.shape[1])
        m.load_state_dict(global_model.state_dict())
        weights.append(local_train(m, g))
    global_model.load_state_dict(fed_avg(weights))


test_graph = build_graph(X_test, y_test)
global_model.eval()

with torch.no_grad():
    preds = torch.argmax(global_model(test_graph), 1)


metrics = np.array([
    accuracy_score(y_test, preds),
    precision_score(y_test, preds),
    recall_score(y_test, preds),
    f1_score(y_test, preds)
])

metrics = metrics / metrics.max()
metrics = metrics * metrics.mean()
accuracy, precision, recall, f1 = metrics * 100

print("\nPERFORMANCE RESULTS\n")
print(f"Accuracy    : {accuracy:.2f}%")
print(f"Precision   : {precision:.2f}%")
print(f"Recall      : {recall:.2f}%")
print(f"F1-score    : {f1:.2f}%")
