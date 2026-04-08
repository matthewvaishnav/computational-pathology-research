"""
Quick demo with minimal training - proves the architecture works fast.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

from src.models import MultimodalFusionModel, ClassificationHead
from experiments.tracking import log_experiment

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('results/quick_demo', exist_ok=True)
os.makedirs('models', exist_ok=True)

class SyntheticMultimodalDataset(Dataset):
    """Synthetic dataset with strong class separation for quick convergence."""
    
    def __init__(self, num_samples=200, num_classes=3, missing_rate=0.1):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.missing_rate = missing_rate
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx].item()
        
        # Strong class-specific patterns for quick learning
        if np.random.rand() > self.missing_rate:
            num_patches = np.random.randint(30, 80)
            wsi_features = torch.randn(num_patches, 1024) + label * 2.0  # Stronger signal
        else:
            wsi_features = None
        
        if np.random.rand() > self.missing_rate:
            genomic = torch.randn(2000) + label * 1.5  # Stronger signal
        else:
            genomic = None
        
        if np.random.rand() > self.missing_rate:
            seq_len = np.random.randint(30, 100)
            clinical_text = torch.randint(1, 30000, (seq_len,))
            clinical_text[:10] = clinical_text[:10] + label * 2000  # Stronger signal
            clinical_text = torch.clamp(clinical_text, 1, 29999)
        else:
            clinical_text = None
        
        return {
            'wsi_features': wsi_features,
            'genomic': genomic,
            'clinical_text': clinical_text,
            'label': self.labels[idx]
        }

def collate_fn(batch):
    batch_size = len(batch)
    labels = torch.stack([item['label'] for item in batch])
    
    # WSI
    wsi_list = [item['wsi_features'] for item in batch]
    max_patches = max((wsi.shape[0] for wsi in wsi_list if wsi is not None), default=0)
    
    if max_patches > 0:
        wsi_padded = torch.zeros(batch_size, max_patches, 1024)
        wsi_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        for i, wsi in enumerate(wsi_list):
            if wsi is not None:
                length = wsi.shape[0]
                wsi_padded[i, :length] = wsi
                wsi_mask[i, :length] = True
    else:
        wsi_padded = None
        wsi_mask = None
    
    # Genomic
    genomic_list = [item['genomic'] for item in batch if item['genomic'] is not None]
    if len(genomic_list) > 0:
        genomic = torch.zeros(batch_size, 2000)
        for i, item in enumerate(batch):
            if item['genomic'] is not None:
                genomic[i] = item['genomic']
    else:
        genomic = None
    
    # Clinical text
    clinical_list = [item['clinical_text'] for item in batch if item['clinical_text'] is not None]
    if len(clinical_list) > 0:
        max_len = max(text.shape[0] for text in clinical_list)
        clinical_text = torch.zeros(batch_size, max_len, dtype=torch.long)
        clinical_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            if item['clinical_text'] is not None:
                length = item['clinical_text'].shape[0]
                clinical_text[i, :length] = item['clinical_text']
                clinical_mask[i, :length] = True
    else:
        clinical_text = None
        clinical_mask = None
    
    return {
        'wsi_features': wsi_padded,
        'wsi_mask': wsi_mask,
        'genomic': genomic,
        'clinical_text': clinical_text,
        'clinical_mask': clinical_mask,
        'label': labels
    }

print("Creating small datasets for quick demo...")
train_dataset = SyntheticMultimodalDataset(num_samples=150, num_classes=3, missing_rate=0.1)
val_dataset = SyntheticMultimodalDataset(num_samples=30, num_classes=3, missing_rate=0.1)
test_dataset = SyntheticMultimodalDataset(num_samples=30, num_classes=3, missing_rate=0.1)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# Smaller model for faster training
print("\nInitializing model...")
model = MultimodalFusionModel(embed_dim=128).to(device)
classifier = ClassificationHead(input_dim=128, num_classes=3).to(device)

total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=5e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

def train_epoch(model, classifier, dataloader, optimizer, criterion, device):
    model.train()
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch.pop('label')
        
        optimizer.zero_grad()
        embeddings = model(batch)
        logits = classifier(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, classifier, dataloader, criterion, device):
    model.eval()
    classifier.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch.pop('label')
            
            embeddings = model(batch)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy, all_preds, all_labels

# Quick training
print("\nStarting quick training (5 epochs)...")
num_epochs = 5
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss, train_acc = train_epoch(model, classifier, train_loader, optimizer, criterion, device)
    val_loss, val_acc, _, _ = evaluate(model, classifier, val_loader, criterion, device)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }, 'models/quick_demo_model.pth')

print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")

# Generate visualizations
print("\nGenerating visualizations...")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/quick_demo/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_curves.png")

# Test evaluation
checkpoint = torch.load('models/quick_demo_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

test_loss, test_acc, test_preds, test_labels = evaluate(model, classifier, test_loader, criterion, device)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(3)],
            yticklabels=[f'Class {i}' for i in range(3)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2%}')
plt.savefig('results/quick_demo/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")

# t-SNE
model.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch_data.pop('label')
        embeddings = model(batch_data)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

# Check for NaN and replace with zeros
if np.isnan(all_embeddings).any():
    print("Warning: Found NaN values in embeddings, replacing with zeros")
    all_embeddings = np.nan_to_num(all_embeddings, nan=0.0)

tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(all_embeddings) - 1))
embeddings_2d = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=all_labels, cmap='viridis', alpha=0.6, s=100)
plt.colorbar(scatter, label='Class')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Multimodal Embeddings')
plt.grid(True, alpha=0.3)
plt.savefig('results/quick_demo/tsne_embeddings.png', dpi=300, bbox_inches='tight')
print("✓ Saved tsne_embeddings.png")

print("\n" + "="*60)
print("QUICK DEMO COMPLETE!")
print("="*60)
print(f"Results:")
print(f"  - Test Accuracy: {test_acc:.4f}")
print(f"  - Best Val Accuracy: {best_val_acc:.4f}")
print(f"\nFiles saved to: results/quick_demo/")

# Log experiment for tracking
config = {
    'num_train_samples': 150,
    'num_val_samples': 30,
    'num_test_samples': 30,
    'num_classes': 3,
    'embed_dim': 128,
    'epochs': 5,
    'batch_size': 16,
    'missing_rate': 0.1
}
metrics = {
    'test_accuracy': float(test_acc),
    'best_val_accuracy': float(best_val_acc),
    'final_train_loss': float(history['train_loss'][-1]),
    'final_train_acc': float(history['train_acc'][-1]),
    'final_val_loss': float(history['val_loss'][-1]),
    'final_val_acc': float(history['val_acc'][-1])
}
exp_path = log_experiment('quick_demo', config, metrics)
print(f"\nExperiment logged to: {exp_path}")
print("="*60)
