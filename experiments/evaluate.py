"""
Evaluation script for trained multimodal fusion models.

This script provides comprehensive evaluation capabilities including:
- Multiple metrics (accuracy, F1, AUC, confusion matrix)
- Per-modality ablation studies
- Missing modality robustness testing
- Attention weight visualization
- Embedding visualization (t-SNE, UMAP)
- Prediction confidence analysis
- Error analysis
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import MultimodalFusionModel, ClassificationHead, SurvivalHead
from src.data import MultimodalDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator.
    
    Provides various evaluation metrics and visualizations for trained models.
    
    Args:
        model: Trained multimodal fusion model
        task_head: Trained task-specific head
        test_loader: Test data loader
        device: Device to run evaluation on
        output_dir: Directory to save results
        config: Evaluation configuration
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_head: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        output_dir: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.task_head = task_head.to(device)
        self.test_loader = test_loader
        self.device = device
        self.config = config or {}
        
        self.output_dir = Path(output_dir) if output_dir else Path('results/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set models to eval mode
        self.model.eval()
        self.task_head.eval()
        
        logger.info(f"Evaluator initialized. Output directory: {self.output_dir}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Basic metrics
        metrics = self.compute_metrics()
        logger.info(f"Metrics: {metrics}")
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Generate visualizations
        if self.config.get('generate_plots', True):
            logger.info("Generating visualizations...")
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.plot_confidence_distribution()
            self.plot_embeddings()
        
        # Ablation studies
        if self.config.get('run_ablation', False):
            logger.info("Running ablation studies...")
            ablation_results = self.run_ablation_study()
            self.save_ablation_results(ablation_results)
        
        # Missing modality analysis
        if self.config.get('test_missing_modalities', False):
            logger.info("Testing missing modality robustness...")
            missing_results = self.test_missing_modalities()
            self.save_missing_modality_results(missing_results)
        
        # Error analysis
        if self.config.get('error_analysis', True):
            logger.info("Performing error analysis...")
            self.error_analysis()
        
        logger.info(f"Evaluation complete! Results saved to {self.output_dir}")
        return metrics
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Computing metrics"):
                batch = self._batch_to_device(batch)
                labels = batch.pop('label')
                
                # Forward pass
                embeddings = self.model(batch)
                logits = self.task_head(embeddings)
                
                # Get predictions
                task_type = self.config.get('task_type', 'classification')
                num_classes = self.config.get('num_classes', 2)
                
                if task_type == 'classification':
                    if num_classes == 2:
                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds = (probs > 0.5).astype(int)
                    else:
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                else:
                    preds = logits.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Store for later use
        self.all_preds = np.array(all_preds)
        self.all_labels = np.array(all_labels)
        self.all_probs = np.array(all_probs) if all_probs else None
        
        # Compute metrics
        metrics = {}
        
        if self.config.get('task_type', 'classification') == 'classification':
            metrics['accuracy'] = accuracy_score(all_labels, all_preds)
            metrics['precision'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # AUC for binary classification
            if self.config.get('num_classes', 2) == 2 and self.all_probs is not None:
                metrics['auc'] = roc_auc_score(all_labels, self.all_probs)
            
            # Per-class metrics
            num_classes = self.config.get('num_classes', 2)
            for i in range(num_classes):
                class_mask = all_labels == i
                if class_mask.sum() > 0:
                    class_acc = (all_preds[class_mask] == i).mean()
                    metrics[f'class_{i}_accuracy'] = class_acc
        
        return metrics
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.all_labels, self.all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Class {i}' for i in range(cm.shape[0])],
            yticklabels=[f'Class {i}' for i in range(cm.shape[0])]
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved confusion matrix to {save_path}")
    
    def plot_roc_curve(self):
        """Plot ROC curve (binary classification only)."""
        if self.config.get('num_classes', 2) != 2 or self.all_probs is None:
            return
        
        fpr, tpr, thresholds = roc_curve(self.all_labels, self.all_probs)
        auc = roc_auc_score(self.all_labels, self.all_probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved ROC curve to {save_path}")
    
    def plot_precision_recall_curve(self):
        """Plot precision-recall curve (binary classification only)."""
        if self.config.get('num_classes', 2) != 2 or self.all_probs is None:
            return
        
        precision, recall, thresholds = precision_recall_curve(self.all_labels, self.all_probs)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved precision-recall curve to {save_path}")
    
    def plot_confidence_distribution(self):
        """Plot prediction confidence distribution."""
        if self.all_probs is None:
            return
        
        # Get confidence (max probability)
        if len(self.all_probs.shape) == 1:
            confidences = np.abs(self.all_probs - 0.5) + 0.5  # Binary case
        else:
            confidences = np.max(self.all_probs, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = self.all_preds == self.all_labels
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([correct_conf, incorrect_conf], labels=['Correct', 'Incorrect'])
        plt.ylabel('Confidence')
        plt.title('Confidence by Correctness')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved confidence distribution to {save_path}")
    
    def plot_embeddings(self):
        """Plot t-SNE visualization of embeddings."""
        logger.info("Computing embeddings for visualization...")
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Extracting embeddings"):
                batch = self._batch_to_device(batch)
                labels = batch.pop('label')
                
                embeddings = self.model(batch)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Handle NaN values
        if np.isnan(all_embeddings).any():
            logger.warning("Found NaN values in embeddings, replacing with zeros")
            all_embeddings = np.nan_to_num(all_embeddings, nan=0.0)
        
        # t-SNE
        logger.info("Computing t-SNE...")
        perplexity = min(30, len(all_embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=all_labels, cmap='viridis', alpha=0.6, s=50
        )
        plt.colorbar(scatter, label='Class')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization of Multimodal Embeddings')
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / 'tsne_embeddings.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved t-SNE visualization to {save_path}")
    
    def run_ablation_study(self) -> Dict[str, Dict[str, float]]:
        """Run ablation study by removing each modality."""
        modalities = ['wsi', 'genomic', 'clinical']
        results = {}
        
        for modality_to_remove in modalities:
            logger.info(f"Testing without {modality_to_remove}...")
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Ablation: no {modality_to_remove}"):
                    batch = self._batch_to_device(batch)
                    labels = batch.pop('label')
                    
                    # Remove modality
                    if modality_to_remove == 'wsi':
                        batch['wsi_features'] = None
                        batch['wsi_mask'] = None
                    elif modality_to_remove == 'genomic':
                        batch['genomic'] = None
                    elif modality_to_remove == 'clinical':
                        batch['clinical_text'] = None
                        batch['clinical_mask'] = None
                    
                    # Forward pass
                    embeddings = self.model(batch)
                    logits = self.task_head(embeddings)
                    
                    # Get predictions
                    if self.config.get('num_classes', 2) == 2:
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                    else:
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            # Compute metrics
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            results[f'without_{modality_to_remove}'] = {
                'accuracy': accuracy,
                'f1': f1
            }
            logger.info(f"Without {modality_to_remove}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return results
    
    def test_missing_modalities(self) -> Dict[str, Dict[str, float]]:
        """Test robustness to missing modalities."""
        missing_rates = [0.0, 0.25, 0.5, 0.75]
        results = {}
        
        for missing_rate in missing_rates:
            logger.info(f"Testing with {missing_rate*100:.0f}% missing data...")
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Missing rate: {missing_rate}"):
                    batch = self._batch_to_device(batch)
                    labels = batch.pop('label')
                    batch_size = labels.shape[0]
                    
                    # Randomly remove modalities
                    if np.random.rand() < missing_rate:
                        batch['wsi_features'] = None
                        batch['wsi_mask'] = None
                    if np.random.rand() < missing_rate:
                        batch['genomic'] = None
                    if np.random.rand() < missing_rate:
                        batch['clinical_text'] = None
                        batch['clinical_mask'] = None
                    
                    # Forward pass
                    embeddings = self.model(batch)
                    logits = self.task_head(embeddings)
                    
                    # Get predictions
                    if self.config.get('num_classes', 2) == 2:
                        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
                    else:
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            # Compute metrics
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            results[f'missing_{int(missing_rate*100)}pct'] = {
                'accuracy': accuracy,
                'f1': f1
            }
            logger.info(f"Missing {missing_rate*100:.0f}%: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        return results
    
    def error_analysis(self):
        """Perform detailed error analysis."""
        # Find misclassified samples
        errors = self.all_preds != self.all_labels
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            logger.info("No errors found!")
            return
        
        # Analyze error patterns
        error_report = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(self.all_labels),
            'errors_by_true_class': {},
            'errors_by_predicted_class': {}
        }
        
        for i in range(self.config.get('num_classes', 2)):
            true_class_errors = errors[self.all_labels == i].sum()
            pred_class_errors = errors[self.all_preds == i].sum()
            
            error_report['errors_by_true_class'][f'class_{i}'] = int(true_class_errors)
            error_report['errors_by_predicted_class'][f'class_{i}'] = int(pred_class_errors)
        
        # Save error report
        report_path = self.output_dir / 'error_analysis.json'
        with open(report_path, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        logger.info(f"✓ Saved error analysis to {report_path}")
        logger.info(f"Total errors: {error_report['total_errors']} ({error_report['error_rate']:.2%})")
    
    def save_metrics(self, metrics: Dict[str, float]):
        """Save metrics to JSON file."""
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Saved metrics to {metrics_path}")
        
        # Also save as text report
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        logger.info(f"✓ Saved evaluation report to {report_path}")
    
    def save_ablation_results(self, results: Dict[str, Dict[str, float]]):
        """Save ablation study results."""
        results_path = self.output_dir / 'ablation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Saved ablation results to {results_path}")
    
    def save_missing_modality_results(self, results: Dict[str, Dict[str, float]]):
        """Save missing modality test results."""
        results_path = self.output_dir / 'missing_modality_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Saved missing modality results to {results_path}")
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained multimodal fusion model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--num-classes', type=int, default=4,
                       help='Number of classes')
    parser.add_argument('--task-type', type=str, default='classification',
                       choices=['classification', 'survival'],
                       help='Task type')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation arguments
    parser.add_argument('--output-dir', type=str, default='./results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--generate-plots', action='store_true', default=True,
                       help='Generate visualization plots')
    parser.add_argument('--run-ablation', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--test-missing-modalities', action='store_true',
                       help='Test missing modality robustness')
    parser.add_argument('--error-analysis', action='store_true', default=True,
                       help='Perform error analysis')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model
    logger.info("Initializing model...")
    model = MultimodalFusionModel(embed_dim=args.embed_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize task head
    if args.task_type == 'classification':
        task_head = ClassificationHead(
            input_dim=args.embed_dim,
            num_classes=args.num_classes
        )
    else:
        task_head = SurvivalHead(input_dim=args.embed_dim)
    
    task_head.load_state_dict(checkpoint['task_head_state_dict'])
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    logger.info(f"Loading {args.split} data...")
    try:
        test_dataset = MultimodalDataset(
            data_dir=args.data_dir,
            split=args.split,
            config=vars(args)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Loaded {len(test_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Please ensure data is available in the specified directory")
        return
    
    # Create config
    config = vars(args)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        task_head=task_head,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        config=config
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
