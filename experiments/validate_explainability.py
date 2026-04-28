"""
Explainability Validation Framework for PCam
Validates that BiomedCLIP explanations are accurate and useful
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.foundation.multi_disease_model import create_foundation_model
from src.explainability.vision_language_explainer import VisionLanguageExplainer
from src.training.train_pcam import PCamDataset


@dataclass
class ExplanationMetrics:
    """Metrics for explanation quality"""
    faithfulness_score: float  # Does explanation match model behavior?
    consistency_score: float  # Similar inputs → similar explanations?
    plausibility_score: float  # Does it make medical sense?
    conciseness_score: float  # Is it concise?
    coverage: float  # What % of predictions have good explanations?
    avg_generation_time: float
    
    def to_dict(self):
        return asdict(self)


class ExplanationValidator:
    """Validates explanation quality"""
    
    def __init__(self, model, explainer, device='cuda'):
        self.model = model
        self.explainer = explainer
        self.device = device
        
    def validate_faithfulness(self, dataloader: DataLoader, num_samples: int = 100) -> float:
        """
        Faithfulness: Does the explanation match model behavior?
        Test by masking important regions and checking prediction change
        """
        faithfulness_scores = []
        
        self.model.eval()
        samples_processed = 0
        
        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc="Testing faithfulness"):
                if samples_processed >= num_samples:
                    break
                    
                patches = patches.to(self.device)
                labels = labels.to(self.device)
                
                # Get original prediction
                output = self.model(patches, disease_type="breast")
                orig_pred = output["breast"]
                orig_prob = F.softmax(orig_pred, dim=1)
                
                # Get explanation with attribution
                explanation = self.explainer.generate_explanation(
                    self.model, patches, {"breast": orig_pred}, 
                    disease_type="breast", return_counterfactual=False
                )
                
                # Get patch-level attribution
                attribution = explanation.feature_attribution["patch_attribution"]
                
                # Mask top 20% most important patches
                batch_size, num_patches = patches.shape[0], patches.shape[1]
                for b in range(batch_size):
                    if samples_processed >= num_samples:
                        break
                        
                    # Get importance scores per patch
                    patch_importance = attribution[b].mean(dim=(1, 2, 3))  # [num_patches]
                    
                    # Mask top 20% patches
                    k = max(1, int(0.2 * num_patches))
                    top_k_indices = torch.topk(patch_importance, k).indices
                    
                    masked_patches = patches[b:b+1].clone()
                    masked_patches[0, top_k_indices] = 0  # Zero out important patches
                    
                    # Get prediction on masked input
                    masked_output = self.model(masked_patches, disease_type="breast")
                    masked_pred = masked_output["breast"]
                    masked_prob = F.softmax(masked_pred, dim=1)
                    
                    # Faithfulness = how much prediction changed
                    prob_change = torch.abs(orig_prob[b] - masked_prob[0]).sum().item()
                    faithfulness_scores.append(prob_change)
                    
                    samples_processed += 1
        
        return np.mean(faithfulness_scores)
    
    def validate_consistency(self, dataloader: DataLoader, num_pairs: int = 50) -> float:
        """
        Consistency: Similar inputs should have similar explanations
        """
        self.model.eval()
        
        # Collect features and explanations
        features_list = []
        explanations_list = []
        
        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc="Collecting for consistency"):
                if len(features_list) >= num_pairs * 2:
                    break
                    
                patches = patches.to(self.device)
                
                # Get features
                output = self.model(patches, disease_type="breast", return_features=True)
                features = output["features"].mean(dim=1)  # Global pool
                
                # Get explanation
                pred = {"breast": output["breast"]}
                explanation = self.explainer.generate_explanation(
                    self.model, patches, pred, disease_type="breast", 
                    return_counterfactual=False
                )
                
                features_list.append(features.cpu())
                explanations_list.append(explanation.natural_language_explanation)
        
        # Compute consistency for similar pairs
        consistency_scores = []
        features_tensor = torch.cat(features_list, dim=0)
        
        for i in range(min(num_pairs, len(features_list) - 1)):
            # Find most similar sample
            similarities = F.cosine_similarity(
                features_tensor[i:i+1], features_tensor[i+1:], dim=1
            )
            most_similar_idx = torch.argmax(similarities).item() + i + 1
            
            # Compare explanations (simple word overlap)
            exp1_words = set(explanations_list[i].lower().split())
            exp2_words = set(explanations_list[most_similar_idx].lower().split())
            
            overlap = len(exp1_words & exp2_words) / len(exp1_words | exp2_words)
            consistency_scores.append(overlap)
        
        return np.mean(consistency_scores)
    
    def validate_plausibility(self, dataloader: DataLoader, num_samples: int = 100) -> float:
        """
        Plausibility: Does explanation contain medically relevant terms?
        """
        medical_terms = {
            'malignant', 'benign', 'tumor', 'cancer', 'metastasis', 'metastatic',
            'ductal', 'lobular', 'carcinoma', 'architecture', 'nuclear', 'atypia',
            'mitotic', 'necrosis', 'invasion', 'invasive', 'tissue', 'cells',
            'glandular', 'stromal', 'lymph', 'node', 'breast', 'pathology'
        }
        
        plausibility_scores = []
        samples_processed = 0
        
        self.model.eval()
        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc="Testing plausibility"):
                if samples_processed >= num_samples:
                    break
                    
                patches = patches.to(self.device)
                
                output = self.model(patches, disease_type="breast")
                pred = {"breast": output["breast"]}
                
                explanation = self.explainer.generate_explanation(
                    self.model, patches, pred, disease_type="breast",
                    return_counterfactual=False
                )
                
                # Count medical terms
                exp_words = set(explanation.natural_language_explanation.lower().split())
                medical_count = len(exp_words & medical_terms)
                
                # Score based on presence of medical terms
                plausibility = min(1.0, medical_count / 5.0)  # Expect ~5 medical terms
                plausibility_scores.append(plausibility)
                
                samples_processed += patches.shape[0]
        
        return np.mean(plausibility_scores)
    
    def validate_conciseness(self, dataloader: DataLoader, num_samples: int = 100) -> float:
        """
        Conciseness: Explanations should be brief but informative
        Target: 50-150 words
        """
        conciseness_scores = []
        samples_processed = 0
        
        self.model.eval()
        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc="Testing conciseness"):
                if samples_processed >= num_samples:
                    break
                    
                patches = patches.to(self.device)
                
                output = self.model(patches, disease_type="breast")
                pred = {"breast": output["breast"]}
                
                explanation = self.explainer.generate_explanation(
                    self.model, patches, pred, disease_type="breast",
                    return_counterfactual=False
                )
                
                word_count = len(explanation.natural_language_explanation.split())
                
                # Optimal range: 50-150 words
                if 50 <= word_count <= 150:
                    score = 1.0
                elif word_count < 50:
                    score = word_count / 50.0
                else:
                    score = max(0.0, 1.0 - (word_count - 150) / 150.0)
                
                conciseness_scores.append(score)
                samples_processed += patches.shape[0]
        
        return np.mean(conciseness_scores)
    
    def validate_coverage(self, dataloader: DataLoader, num_samples: int = 100) -> float:
        """
        Coverage: What % of predictions get reasonable explanations?
        """
        successful = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc="Testing coverage"):
                if total >= num_samples:
                    break
                    
                patches = patches.to(self.device)
                
                try:
                    output = self.model(patches, disease_type="breast")
                    pred = {"breast": output["breast"]}
                    
                    explanation = self.explainer.generate_explanation(
                        self.model, patches, pred, disease_type="breast",
                        return_counterfactual=False
                    )
                    
                    # Check if explanation is reasonable (>20 words, contains medical terms)
                    exp_text = explanation.natural_language_explanation
                    if len(exp_text.split()) > 20 and any(term in exp_text.lower() 
                                                           for term in ['malignant', 'benign', 'tissue', 'cancer']):
                        successful += 1
                    
                except Exception as e:
                    print(f"Failed to generate explanation: {e}")
                
                total += patches.shape[0]
        
        return successful / total if total > 0 else 0.0
    
    def measure_generation_time(self, dataloader: DataLoader, num_samples: int = 50) -> float:
        """Measure average explanation generation time"""
        times = []
        samples_processed = 0
        
        self.model.eval()
        with torch.no_grad():
            for patches, labels in tqdm(dataloader, desc="Measuring time"):
                if samples_processed >= num_samples:
                    break
                    
                patches = patches.to(self.device)
                
                output = self.model(patches, disease_type="breast")
                pred = {"breast": output["breast"]}
                
                explanation = self.explainer.generate_explanation(
                    self.model, patches, pred, disease_type="breast",
                    return_counterfactual=False
                )
                
                times.append(explanation.explanation_generation_time)
                samples_processed += patches.shape[0]
        
        return np.mean(times)
    
    def run_full_validation(self, dataloader: DataLoader, num_samples: int = 100) -> ExplanationMetrics:
        """Run all validation tests"""
        print("Running explainability validation...")
        
        faithfulness = self.validate_faithfulness(dataloader, num_samples)
        print(f"✓ Faithfulness: {faithfulness:.3f}")
        
        consistency = self.validate_consistency(dataloader, num_samples // 2)
        print(f"✓ Consistency: {consistency:.3f}")
        
        plausibility = self.validate_plausibility(dataloader, num_samples)
        print(f"✓ Plausibility: {plausibility:.3f}")
        
        conciseness = self.validate_conciseness(dataloader, num_samples)
        print(f"✓ Conciseness: {conciseness:.3f}")
        
        coverage = self.validate_coverage(dataloader, num_samples)
        print(f"✓ Coverage: {coverage:.3f}")
        
        avg_time = self.measure_generation_time(dataloader, num_samples // 2)
        print(f"✓ Avg generation time: {avg_time:.3f}s")
        
        return ExplanationMetrics(
            faithfulness_score=faithfulness,
            consistency_score=consistency,
            plausibility_score=plausibility,
            conciseness_score=conciseness,
            coverage=coverage,
            avg_generation_time=avg_time
        )


def visualize_results(metrics: ExplanationMetrics, output_dir: Path):
    """Visualize validation results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Bar chart of metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Faithfulness', 'Consistency', 'Plausibility', 'Conciseness', 'Coverage']
    metric_values = [
        metrics.faithfulness_score,
        metrics.consistency_score,
        metrics.plausibility_score,
        metrics.conciseness_score,
        metrics.coverage
    ]
    
    colors = ['green' if v >= 0.7 else 'orange' if v >= 0.5 else 'red' for v in metric_values]
    
    ax.barh(metric_names, metric_values, color=colors)
    ax.set_xlabel('Score')
    ax.set_title('Explainability Validation Metrics')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(metric_values):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'explainability_metrics.png', dpi=300)
    plt.close()
    
    print(f"Saved visualization to {output_dir / 'explainability_metrics.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='data/pcam_real')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=str, default='results/explainability_validation')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = create_foundation_model(
        encoder_type='resnet18',
        supported_diseases=['breast']
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create explainer
    print("Initializing explainer...")
    explainer = VisionLanguageExplainer(device=device)
    
    # Load validation data
    print("Loading validation data...")
    val_dataset = PCamDataset(
        root_dir=args.data_root,
        split='valid',
        patches_per_sample=50
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Run validation
    validator = ExplanationValidator(model, explainer, device)
    metrics = validator.run_full_validation(val_loader, args.num_samples)
    
    # Save results
    results = {
        'metrics': metrics.to_dict(),
        'checkpoint': args.checkpoint,
        'num_samples': args.num_samples
    }
    
    with open(output_dir / 'explainability_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'explainability_results.json'}")
    
    # Visualize
    visualize_results(metrics, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPLAINABILITY VALIDATION SUMMARY")
    print("="*60)
    print(f"Faithfulness:  {metrics.faithfulness_score:.3f} {'✓' if metrics.faithfulness_score >= 0.7 else '✗'}")
    print(f"Consistency:   {metrics.consistency_score:.3f} {'✓' if metrics.consistency_score >= 0.7 else '✗'}")
    print(f"Plausibility:  {metrics.plausibility_score:.3f} {'✓' if metrics.plausibility_score >= 0.7 else '✗'}")
    print(f"Conciseness:   {metrics.conciseness_score:.3f} {'✓' if metrics.conciseness_score >= 0.7 else '✗'}")
    print(f"Coverage:      {metrics.coverage:.3f} {'✓' if metrics.coverage >= 0.9 else '✗'}")
    print(f"Avg Time:      {metrics.avg_generation_time:.3f}s {'✓' if metrics.avg_generation_time < 5.0 else '✗'}")
    print("="*60)


if __name__ == '__main__':
    main()
