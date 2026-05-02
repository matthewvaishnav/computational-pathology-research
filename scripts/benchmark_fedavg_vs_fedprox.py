"""
FedAvg vs FedProx benchmark on synthetic non-IID pathology data.

Simulates 5 hospital clients with heterogeneous class distributions,
runs 20 federated rounds, and plots convergence comparison.
"""

import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.federated.common.data_models import ClientUpdate
from src.federated.aggregator.fedavg import FedAvgAggregator
from src.federated.aggregator.fedprox import FedProxAggregator

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

NUM_CLIENTS = 5
NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR = 0.01
NUM_CLASSES = 2
INPUT_DIM = 64  # simulated patch features


# ── Model ────────────────────────────────────────────────────────────────────

class PatchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 32), nn.ReLU(), nn.Linear(32, NUM_CLASSES)
        )
    def forward(self, x):
        return self.net(x)


# ── Non-IID data generation ───────────────────────────────────────────────────

def make_noniid_data(num_clients=NUM_CLIENTS, n_per_client=200):
    """Each hospital has a skewed class distribution (simulates real-world heterogeneity)."""
    # Class probabilities per client: ranges from 90% class-0 to 90% class-1
    class_probs = np.linspace(0.9, 0.1, num_clients)
    datasets = []
    for p in class_probs:
        labels = torch.bernoulli(torch.full((n_per_client,), 1 - p)).long()
        # Features: class-conditional Gaussian (different means per client = covariate shift)
        features = torch.randn(n_per_client, INPUT_DIM)
        features += labels.float().unsqueeze(1) * 0.5
        datasets.append((features, labels))
    return datasets


# ── Local training ────────────────────────────────────────────────────────────

def local_train(model, data, global_state, mu=0.0):
    """Train locally. mu>0 adds FedProx proximal term."""
    features, labels = data
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if mu > 0.0:
                # Proximal term: penalise drift from global model
                prox = sum(
                    ((p - g) ** 2).sum()
                    for p, g in zip(model.parameters(), global_state)
                )
                loss = loss + (mu / 2) * prox
            loss.backward()
            optimizer.step()

    # Return gradients (local_state - global_state)
    gradients = {
        name: (param.data - g)
        for (name, param), g in zip(model.named_parameters(), global_state)
    }
    return gradients


def evaluate(model, datasets):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in datasets:
            preds = model(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


# ── Federated round ───────────────────────────────────────────────────────────

def run_experiment(aggregator, datasets, mu=0.0, label=""):
    global_model = PatchClassifier()
    global_state = [p.data.clone() for p in global_model.parameters()]
    accuracies = []

    for round_idx in range(NUM_ROUNDS):
        updates = []
        for client_id, data in enumerate(datasets):
            local_model = PatchClassifier()
            local_model.load_state_dict(global_model.state_dict())
            grads = local_train(local_model, data, global_state, mu=mu)
            updates.append(ClientUpdate(
                client_id=str(client_id),
                round_id=round_idx,
                gradients=grads,
                dataset_size=len(data[0]),
            ))

        agg = aggregator.aggregate(updates)

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in agg:
                    param.data += agg[name]

        global_state = [p.data.clone() for p in global_model.parameters()]
        acc = evaluate(global_model, datasets)
        accuracies.append(acc)
        print(f"[{label}] Round {round_idx+1:2d}/{NUM_ROUNDS}  acc={acc:.3f}")

    return accuracies


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    datasets = make_noniid_data()

    print("\n=== FedAvg ===")
    fedavg_acc = run_experiment(FedAvgAggregator(), datasets, mu=0.0, label="FedAvg")

    # Reset seed so both start from same point
    torch.manual_seed(SEED); np.random.seed(SEED)

    print("\n=== FedProx (μ=0.01) ===")
    fedprox_acc = run_experiment(FedProxAggregator(mu=0.01), datasets, mu=0.01, label="FedProx")

    # ── Plot ──────────────────────────────────────────────────────────────────
    rounds = range(1, NUM_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, fedavg_acc,  "o-", label="FedAvg",        color="#2196F3", linewidth=2)
    ax.plot(rounds, fedprox_acc, "s-", label="FedProx (μ=0.01)", color="#E91E63", linewidth=2)
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Global Accuracy", fontsize=12)
    ax.set_title("FedAvg vs FedProx — Non-IID Synthetic Pathology Data\n(5 hospitals, heterogeneous class distributions)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    final_fedavg  = fedavg_acc[-1]
    final_fedprox = fedprox_acc[-1]
    ax.annotate(f"{final_fedavg:.1%}", xy=(NUM_ROUNDS, final_fedavg),
                xytext=(-30, 8), textcoords="offset points", color="#2196F3", fontsize=10)
    ax.annotate(f"{final_fedprox:.1%}", xy=(NUM_ROUNDS, final_fedprox),
                xytext=(-30, -14), textcoords="offset points", color="#E91E63", fontsize=10)

    out = "results/fedavg_vs_fedprox.png"
    import os; os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")

    print(f"\nFinal accuracy — FedAvg: {final_fedavg:.1%}  |  FedProx: {final_fedprox:.1%}")
    diff = final_fedprox - final_fedavg
    if abs(diff) < 0.005:
        print("Verdict: negligible difference on this data distribution.")
    elif diff > 0:
        print(f"Verdict: FedProx +{diff:.1%} — proximal term helped with non-IID drift.")
    else:
        print(f"Verdict: FedAvg +{-diff:.1%} — FedProx overhead not justified here.")
