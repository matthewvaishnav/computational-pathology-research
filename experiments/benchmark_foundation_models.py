"""
Benchmark foundation models for computational pathology.

Compares UNI, Phikon, GigaPath, CTransPath, and ResNet50 baselines on:
- Feature extraction speed
- Memory usage
- Classification performance
- Feature quality metrics

Usage:
    python experiments/benchmark_foundation_models.py \
        --data-dir data/processed \
        --models uni phikon resnet50_imagenet \
        --output results/foundation_model_benchmark.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.data.datasets import PCamDataset
from src.models.pretrained import PRETRAINED_MODELS, PretrainedFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FoundationModelBenchmark:
    """Benchmark suite for foundation models."""
    
    def __init__(
        self,
        data_dir: Path,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load datasets
        logger.info("Loading datasets...")
        self.train_dataset = PCamDataset(data_dir / "train", transform=None)
        self.val_dataset = PCamDataset(data_dir / "val", transform=None)
        
        # Limit to subset for faster benchmarking
        self.train_dataset = torch.utils.data.Subset(
            self.train_dataset, range(min(10000, len(self.train_dataset)))
        )
        self.val_dataset = torch.utils.data.Subset(
            self.val_dataset, range(min(2000, len(self.val_dataset)))
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def benchmark_model(self, model_name: str) -> Dict:
        """Run complete benchmark for a single model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"{'='*60}")
        
        results = {
            "model_name": model_name,
            "config": PRETRAINED_MODELS.get(model_name, {}),
        }
        
        try:
            # Load model
            logger.info("Loading model...")
            start_time = time.time()
            extractor = PretrainedFeatureExtractor(
                model_name,
                freeze=True,
                device=self.device,
            )
            load_time = time.time() - start_time
            results["load_time_sec"] = load_time
            
            # Benchmark inference speed
            logger.info("Benchmarking inference speed...")
            speed_results = self._benchmark_speed(extractor)
            results.update(speed_results)
            
            # Benchmark memory usage
            logger.info("Benchmarking memory usage...")
            memory_results = self._benchmark_memory(extractor)
            results.update(memory_results)
            
            # Extract features
            logger.info("Extracting features...")
            train_features, train_labels = self._extract_features(
                extractor, self.train_dataset
            )
            val_features, val_labels = self._extract_features(
                extractor, self.val_dataset
            )
            
            # Benchmark classification performance
            logger.info("Benchmarking classification...")
            clf_results = self._benchmark_classification(
                train_features, train_labels,
                val_features, val_labels,
            )
            results.update(clf_results)
            
            # Feature quality metrics
            logger.info("Computing feature quality metrics...")
            quality_results = self._compute_feature_quality(
                train_features, train_labels
            )
            results.update(quality_results)
            
            results["status"] = "success"
            
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _benchmark_speed(self, extractor: nn.Module) -> Dict:
        """Benchmark inference speed."""
        # Create dummy batch
        dummy_input = torch.randn(
            self.batch_size, 3, 224, 224,
            device=self.device
        )
        
        # Warmup
        for _ in range(10):
            _ = extractor(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        num_iterations = 100
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = extractor(dummy_input)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        return {
            "inference_time_ms": (elapsed / num_iterations) * 1000,
            "throughput_samples_per_sec": (self.batch_size * num_iterations) / elapsed,
        }
    
    def _benchmark_memory(self, extractor: nn.Module) -> Dict:
        """Benchmark memory usage."""
        if self.device != "cuda":
            return {"gpu_memory_mb": 0}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Run inference
        dummy_input = torch.randn(
            self.batch_size, 3, 224, 224,
            device=self.device
        )
        
        with torch.no_grad():
            _ = extractor(dummy_input)
        
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {"gpu_memory_mb": peak_memory}
    
    def _extract_features(
        self,
        extractor: nn.Module,
        dataset: torch.utils.data.Dataset,
    ) -> tuple:
        """Extract features for entire dataset."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        all_features = []
        all_labels = []
        
        extractor.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting"):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    labels = batch["label"]
                else:
                    images, labels = batch
                    images = images.to(self.device)
                
                features = extractor(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())
        
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return features, labels
    
    def _benchmark_classification(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
    ) -> Dict:
        """Benchmark linear classification performance."""
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features, train_labels)
        
        # Predict
        val_preds = clf.predict(val_features)
        val_probs = clf.predict_proba(val_features)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_probs)
        
        return {
            "linear_probe_accuracy": float(accuracy),
            "linear_probe_auc": float(auc),
        }
    
    def _compute_feature_quality(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """Compute feature quality metrics."""
        # Feature statistics
        feature_mean = float(np.mean(features))
        feature_std = float(np.std(features))
        feature_norm = float(np.linalg.norm(features, axis=1).mean())
        
        # Intra-class vs inter-class distance
        class_0_features = features[labels == 0]
        class_1_features = features[labels == 1]
        
        intra_class_dist = float(
            np.mean([
                np.linalg.norm(class_0_features - class_0_features.mean(axis=0), axis=1).mean(),
                np.linalg.norm(class_1_features - class_1_features.mean(axis=0), axis=1).mean(),
            ])
        )
        
        inter_class_dist = float(
            np.linalg.norm(
                class_0_features.mean(axis=0) - class_1_features.mean(axis=0)
            )
        )
        
        separability = inter_class_dist / (intra_class_dist + 1e-8)
        
        return {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_norm": feature_norm,
            "intra_class_distance": intra_class_dist,
            "inter_class_distance": inter_class_dist,
            "separability_ratio": float(separability),
        }
    
    def run_benchmark(self, model_names: List[str]) -> Dict:
        """Run benchmark for multiple models."""
        results = {
            "benchmark_config": {
                "data_dir": str(self.data_dir),
                "device": self.device,
                "batch_size": self.batch_size,
                "train_samples": len(self.train_dataset),
                "val_samples": len(self.val_dataset),
            },
            "models": {},
        }
        
        for model_name in model_names:
            if model_name not in PRETRAINED_MODELS:
                logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            model_results = self.benchmark_model(model_name)
            results["models"][model_name] = model_results
        
        return results


def print_summary(results: Dict) -> None:
    """Print benchmark summary table."""
    print("\n" + "="*80)
    print("Foundation Model Benchmark Summary")
    print("="*80)
    
    # Table header
    print(f"\n{'Model':<20} {'Speed (ms)':<15} {'Memory (MB)':<15} {'Accuracy':<12} {'AUC':<12}")
    print("-"*80)
    
    # Table rows
    for model_name, model_results in results["models"].items():
        if model_results["status"] != "success":
            print(f"{model_name:<20} {'FAILED':<15}")
            continue
        
        speed = model_results.get("inference_time_ms", 0)
        memory = model_results.get("gpu_memory_mb", 0)
        accuracy = model_results.get("linear_probe_accuracy", 0)
        auc = model_results.get("linear_probe_auc", 0)
        
        print(
            f"{model_name:<20} "
            f"{speed:<15.2f} "
            f"{memory:<15.1f} "
            f"{accuracy:<12.4f} "
            f"{auc:<12.4f}"
        )
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark foundation models")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["uni", "phikon", "resnet50_imagenet"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/foundation_model_benchmark.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Run benchmark
    benchmark = FoundationModelBenchmark(
        Path(args.data_dir),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    results = benchmark.run_benchmark(args.models)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
