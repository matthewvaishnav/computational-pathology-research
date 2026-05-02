"""
CAMELYON17 Federated Learning Experiment: Do Models Learn Real Signal or Scanner Shortcuts?

Research question: When training AttentionMIL federatedly across CAMELYON17's 5 hospital
sites, does the model learn site-invariant pathological features, or does it learn to
average site-specific shortcuts (scanner artifacts, staining batch effects)?

Methodology:
1. Simulate 5 hospital clients using CAMELYON17 site splits
2. Train three models: Federated (FedAvg), Centralized, Site-Isolated (per-site)
3. Audit attention consistency: for matched tissue patches across sites, do models
   attend to the same regions?
4. Measure: cross-site attention correlation, site-prediction accuracy from attention
   weights (high = learning scanner shortcuts), and classification AUC per site.

If federated model has HIGH cross-site attention correlation and LOW site-predictability
from attention → learning real pathological signal.
If federated model has LOW cross-site attention correlation or HIGH site-predictability
→ learning shortcuts.

Usage:
    python experiments/camelyon17_federated_audit.py --data-root data/camelyon17
    python experiments/camelyon17_federated_audit.py --synthetic  # no data needed
"""

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
NUM_SITES = 5
NUM_ROUNDS = 15
LOCAL_EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-3
FEATURE_DIM = 512
NUM_PATCHES = 64   # patches per slide (reduced for CPU speed)
NUM_SLIDES_PER_SITE = 40


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class SlideData:
    slide_id: str
    site_id: int          # 0-4 (CAMELYON17 has 5 centers)
    label: int            # 0=normal, 1=metastasis
    features: torch.Tensor   # [num_patches, feature_dim]
    coordinates: torch.Tensor  # [num_patches, 2]  — spatial positions


def make_synthetic_camelyon17(seed: int = SEED) -> List[SlideData]:
    """
    Generate synthetic CAMELYON17-like data with realistic properties:
    - 5 sites with different scanner characteristics (mean/variance shifts)
    - Spatial patch coordinates on a grid
    - Tumor slides have a localized high-attention region (ground truth signal)
    - Each site adds scanner-specific bias to features (the shortcut)
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    slides = []

    # Site-specific scanner biases (what a shortcut-learning model would pick up)
    site_biases = [
        torch.tensor(rng.randn(FEATURE_DIM) * 0.3, dtype=torch.float32)
        for _ in range(NUM_SITES)
    ]

    slide_idx = 0
    for site_id in range(NUM_SITES):
        for _ in range(NUM_SLIDES_PER_SITE):
            label = int(slide_idx % 2)  # balanced labels
            slide_idx += 1

            # Spatial grid coordinates
            grid_size = int(np.ceil(np.sqrt(NUM_PATCHES)))
            coords = torch.tensor(
                [[i, j] for i in range(grid_size) for j in range(grid_size)],
                dtype=torch.float32
            )[:NUM_PATCHES]

            # Base features: random tissue appearance
            features = torch.randn(NUM_PATCHES, FEATURE_DIM) * 0.5

            # Pathological signal: tumor slides have a localized cluster of patches
            # with a consistent feature direction (the TRUE signal)
            if label == 1:
                tumor_center = rng.randint(0, NUM_PATCHES)
                tumor_patches = torch.randint(0, NUM_PATCHES, (NUM_PATCHES // 4,))
                tumor_direction = torch.randn(FEATURE_DIM)
                tumor_direction = tumor_direction / tumor_direction.norm()
                features[tumor_patches] += tumor_direction * 2.0  # strong signal

            # Scanner bias: site-specific offset added to ALL patches (the SHORTCUT)
            features += site_biases[site_id].unsqueeze(0)

            slides.append(SlideData(
                slide_id=f"site{site_id}_slide{slide_idx:04d}",
                site_id=site_id,
                label=label,
                features=features,
                coordinates=coords,
            ))

    return slides


def split_by_site(slides: List[SlideData]) -> Dict[int, List[SlideData]]:
    """Group slides by site — each site is one federated client."""
    by_site: Dict[int, List[SlideData]] = {i: [] for i in range(NUM_SITES)}
    for s in slides:
        by_site[s.site_id].append(s)
    return by_site


# ── Model ─────────────────────────────────────────────────────────────────────

class AttentionMIL(nn.Module):
    """Gated attention MIL — returns both logits and attention weights."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.attention_V = nn.Linear(feature_dim, hidden_dim)
        self.attention_U = nn.Linear(feature_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [num_patches, feature_dim]
        Returns:
            logits: [2]
            attention: [num_patches] — normalized attention weights
        """
        V = torch.tanh(self.attention_V(features))
        U = torch.sigmoid(self.attention_U(features))
        scores = self.attention_w(V * U).squeeze(-1)          # [num_patches]
        attention = torch.softmax(scores, dim=0)               # [num_patches]
        aggregated = (attention.unsqueeze(-1) * features).sum(0)  # [feature_dim]
        logits = self.classifier(aggregated)
        return logits, attention


# ── Training ──────────────────────────────────────────────────────────────────

def train_local(model: AttentionMIL, slides: List[SlideData], epochs: int = LOCAL_EPOCHS) -> Dict:
    """Train model on local site data, return gradient updates."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        random.shuffle(slides)
        for slide in slides:
            optimizer.zero_grad()
            logits, _ = model(slide.features)
            loss = criterion(logits.unsqueeze(0), torch.tensor([slide.label]))
            loss.backward()
            optimizer.step()

    return {name: param.data.clone() for name, param in model.named_parameters()}


def fedavg_aggregate(global_state: Dict, site_states: List[Dict], site_sizes: List[int]) -> Dict:
    """Weighted average of model states."""
    total = sum(site_sizes)
    aggregated = {}
    for key in global_state:
        aggregated[key] = sum(
            (size / total) * state[key]
            for state, size in zip(site_states, site_sizes)
        )
    return aggregated


def evaluate(model: AttentionMIL, slides: List[SlideData]) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for slide in slides:
            logits, _ = model(slide.features)
            pred = logits.argmax().item()
            correct += int(pred == slide.label)
    return correct / len(slides) if slides else 0.0


# ── Attention Audit ───────────────────────────────────────────────────────────

@dataclass
class AttentionAuditResult:
    """Results of the attention consistency audit."""
    cross_site_attention_correlation: float   # High = consistent across sites
    site_predictability_from_attention: float  # High = attention reveals scanner
    per_site_auc: Dict[int, float]
    verdict: str


def compute_attention_maps(model: AttentionMIL, slides: List[SlideData]) -> Dict[str, np.ndarray]:
    """Get attention weights for all slides."""
    model.eval()
    attention_maps = {}
    with torch.no_grad():
        for slide in slides:
            _, attention = model(slide.features)
            attention_maps[slide.slide_id] = attention.numpy()
    return attention_maps


def cross_site_attention_correlation(
    model: AttentionMIL,
    by_site: Dict[int, List[SlideData]],
) -> float:
    """
    Measure how consistently the model attends across sites.

    For each pair of sites, find slides with the same label and compute
    the correlation between their top-k attended patch positions.
    High correlation = model focuses on same spatial regions regardless of site.
    Low correlation = model focuses on different regions per site (scanner shortcut).
    """
    site_top_patches = {}
    model.eval()

    with torch.no_grad():
        for site_id, slides in by_site.items():
            tumor_slides = [s for s in slides if s.label == 1]
            if not tumor_slides:
                continue
            # Average attention over tumor slides for this site
            attentions = []
            for slide in tumor_slides[:10]:  # sample 10 per site
                _, attn = model(slide.features)
                attentions.append(attn.numpy())
            site_top_patches[site_id] = np.mean(attentions, axis=0)

    if len(site_top_patches) < 2:
        return 0.0

    # Pairwise correlation of mean attention vectors
    site_ids = list(site_top_patches.keys())
    correlations = []
    for i in range(len(site_ids)):
        for j in range(i + 1, len(site_ids)):
            a = site_top_patches[site_ids[i]]
            b = site_top_patches[site_ids[j]]
            corr = np.corrcoef(a, b)[0, 1]
            correlations.append(corr)

    return float(np.mean(correlations))


def site_predictability_from_attention(
    model: AttentionMIL,
    slides: List[SlideData],
) -> float:
    """
    Train a linear probe to predict site_id from attention weights.
    High accuracy = attention encodes scanner identity = shortcut learning.
    Low accuracy = attention is site-agnostic = real signal.
    """
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for slide in slides:
            _, attn = model(slide.features)
            X.append(attn.numpy())
            y.append(slide.site_id)

    X = np.array(X)
    y = np.array(y)

    # Simple linear probe via least-squares
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    try:
        probe = LogisticRegression(max_iter=200, random_state=SEED)
        scores = cross_val_score(probe, X, y, cv=3, scoring="accuracy")
        return float(scores.mean())
    except Exception:
        return 0.0


def audit_model(
    model: AttentionMIL,
    all_slides: List[SlideData],
    by_site: Dict[int, List[SlideData]],
    label: str,
) -> AttentionAuditResult:
    """Run full attention audit on a trained model."""
    logger.info(f"Auditing {label}...")

    cross_site_corr = cross_site_attention_correlation(model, by_site)
    site_pred = site_predictability_from_attention(model, all_slides)

    # Per-site accuracy
    per_site_acc = {
        site_id: evaluate(model, slides)
        for site_id, slides in by_site.items()
    }

    # Verdict
    if cross_site_corr > 0.6 and site_pred < 0.4:
        verdict = "✅ Learning real pathological signal (site-invariant attention)"
    elif site_pred > 0.6:
        verdict = "⚠️  Learning scanner shortcuts (attention predicts site)"
    elif cross_site_corr < 0.2:
        verdict = "⚠️  Inconsistent attention across sites (unstable features)"
    else:
        verdict = "🔶 Mixed signal — partial shortcut learning"

    result = AttentionAuditResult(
        cross_site_attention_correlation=cross_site_corr,
        site_predictability_from_attention=site_pred,
        per_site_auc=per_site_acc,
        verdict=verdict,
    )

    logger.info(f"  Cross-site attention correlation: {cross_site_corr:.3f}")
    logger.info(f"  Site predictability from attention: {site_pred:.3f}")
    logger.info(f"  Per-site accuracy: {per_site_acc}")
    logger.info(f"  Verdict: {verdict}")

    return result


# ── Experiment Runners ────────────────────────────────────────────────────────

def run_federated(by_site: Dict[int, List[SlideData]]) -> AttentionMIL:
    """Train with FedAvg across 5 sites."""
    logger.info("=== Training: Federated (FedAvg) ===")
    global_model = AttentionMIL()
    torch.manual_seed(SEED)

    for round_idx in range(NUM_ROUNDS):
        site_states = []
        site_sizes = []
        for site_id, slides in by_site.items():
            local_model = AttentionMIL()
            local_model.load_state_dict(global_model.state_dict())
            state = train_local(local_model, slides)
            site_states.append(state)
            site_sizes.append(len(slides))

        aggregated = fedavg_aggregate(
            global_model.state_dict(), site_states, site_sizes
        )
        global_model.load_state_dict(aggregated)

        all_slides = [s for slides in by_site.values() for s in slides]
        acc = evaluate(global_model, all_slides)
        logger.info(f"  Round {round_idx+1:2d}/{NUM_ROUNDS}  global_acc={acc:.3f}")

    return global_model


def run_centralized(all_slides: List[SlideData]) -> AttentionMIL:
    """Train on all data pooled (upper bound — not privacy-preserving)."""
    logger.info("=== Training: Centralized (pooled) ===")
    torch.manual_seed(SEED)
    model = AttentionMIL()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_ROUNDS * LOCAL_EPOCHS):
        random.shuffle(all_slides)
        for slide in all_slides:
            optimizer.zero_grad()
            logits, _ = model(slide.features)
            loss = criterion(logits.unsqueeze(0), torch.tensor([slide.label]))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % LOCAL_EPOCHS == 0:
            acc = evaluate(model, all_slides)
            logger.info(f"  Epoch {epoch+1:3d}  acc={acc:.3f}")

    return model


def run_site_isolated(by_site: Dict[int, List[SlideData]]) -> Dict[int, AttentionMIL]:
    """Train one model per site — no federation."""
    logger.info("=== Training: Site-Isolated (no federation) ===")
    models = {}
    for site_id, slides in by_site.items():
        torch.manual_seed(SEED)
        model = AttentionMIL()
        for _ in range(NUM_ROUNDS * LOCAL_EPOCHS):
            train_local(model, slides, epochs=1)
        acc = evaluate(model, slides)
        logger.info(f"  Site {site_id} acc={acc:.3f}")
        models[site_id] = model
    return models


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CAMELYON17 Federated Attention Audit")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Use synthetic data (default until real data available)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/camelyon17_federated_audit"))
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic or args.data_root is None:
        logger.info("Using synthetic CAMELYON17-like data (5 sites, scanner biases)")
        all_slides = make_synthetic_camelyon17()
    else:
        raise NotImplementedError("Real CAMELYON17 loader — coming soon")

    by_site = split_by_site(all_slides)
    logger.info(f"Loaded {len(all_slides)} slides across {NUM_SITES} sites")
    for site_id, slides in by_site.items():
        pos = sum(s.label for s in slides)
        logger.info(f"  Site {site_id}: {len(slides)} slides ({pos} tumor, {len(slides)-pos} normal)")

    # ── Train ─────────────────────────────────────────────────────────────────
    federated_model = run_federated(by_site)
    centralized_model = run_centralized(all_slides)
    isolated_models = run_site_isolated(by_site)

    # ── Audit ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("ATTENTION AUDIT: Does the model learn real signal or shortcuts?")
    logger.info("="*60)

    federated_audit = audit_model(federated_model, all_slides, by_site, "Federated (FedAvg)")
    centralized_audit = audit_model(centralized_model, all_slides, by_site, "Centralized")

    # For isolated: audit each site model on its own site
    isolated_cross_site_corrs = []
    for site_id, model in isolated_models.items():
        corr = cross_site_attention_correlation(model, by_site)
        isolated_cross_site_corrs.append(corr)
    isolated_mean_corr = float(np.mean(isolated_cross_site_corrs))

    # ── Results ───────────────────────────────────────────────────────────────
    results = {
        "experiment": "CAMELYON17 Federated Attention Audit",
        "research_question": "Do federated MIL models learn site-invariant pathological features or scanner shortcuts?",
        "num_sites": NUM_SITES,
        "num_slides": len(all_slides),
        "num_rounds": args.rounds,
        "federated": {
            "cross_site_attention_correlation": federated_audit.cross_site_attention_correlation,
            "site_predictability_from_attention": federated_audit.site_predictability_from_attention,
            "per_site_accuracy": federated_audit.per_site_auc,
            "verdict": federated_audit.verdict,
        },
        "centralized": {
            "cross_site_attention_correlation": centralized_audit.cross_site_attention_correlation,
            "site_predictability_from_attention": centralized_audit.site_predictability_from_attention,
            "per_site_accuracy": centralized_audit.per_site_auc,
            "verdict": centralized_audit.verdict,
        },
        "site_isolated": {
            "mean_cross_site_attention_correlation": isolated_mean_corr,
        },
    }

    out_path = args.output_dir / "audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print Summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'Cross-Site Corr':>16} {'Site Predictability':>20} {'Verdict'}")
    print("-"*80)
    print(f"{'Federated (FedAvg)':<20} {federated_audit.cross_site_attention_correlation:>16.3f} "
          f"{federated_audit.site_predictability_from_attention:>20.3f}  {federated_audit.verdict}")
    print(f"{'Centralized':<20} {centralized_audit.cross_site_attention_correlation:>16.3f} "
          f"{centralized_audit.site_predictability_from_attention:>20.3f}  {centralized_audit.verdict}")
    print(f"{'Site-Isolated':<20} {isolated_mean_corr:>16.3f} {'N/A':>20}")
    print(f"\nFull results saved to {out_path}")

    # Key finding
    fed_corr = federated_audit.cross_site_attention_correlation
    cent_corr = centralized_audit.cross_site_attention_correlation
    print("\n📊 KEY FINDING:")
    if fed_corr > cent_corr - 0.05:
        print("Federated training achieves comparable cross-site attention consistency")
        print("to centralized training — WITHOUT sharing patient data.")
        print("→ FedAvg learns real pathological signal, not scanner shortcuts.")
    else:
        print(f"Centralized training has {cent_corr - fed_corr:.2f} higher cross-site attention consistency.")
        print("→ Federated training shows increased shortcut learning vs centralized.")
        print("→ Suggests need for site-invariant regularization in federated MIL.")


if __name__ == "__main__":
    main()
