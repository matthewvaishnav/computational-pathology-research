"""
Run the working demo to generate actual training results.
This proves the architecture works and generates portfolio-worthy outputs.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

# Import models
from src.models import MultimodalFusionModel, ClassificationHead

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

class SyntheticMultimodalDataset(Dataset):
    """Synthetic dataset mimicking multimodal pathology data."""
    
    def __init__(self, num_samples=1000, num_classes=4, missing_rate=0.2):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.missing_rate = missing_rate
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        label = self.labels[idx].item()
        
        # WSI features with class-specific patterns
        if np.random.rand() > self.missing_rate:
            num_patches = np.random.randint(50, 150)
            wsi_features = torch.randn(num_patches, 1024) + label * 0.5
        else:
            wsi_features = None
        
        # Genomic features
        if np.random.rand() > self.missing_rate:
            genomic = torch.randn(2000) + label * 0.3
        else:
            genomic = None
        
        # Clinical text
        if np.random.rand() > self.missing_rate:
            seq_len = np.random.randint(50, 200)
            clinical_text = torch.randint(1, 30000, (seq_len,))
            clinical_text[:10] = clinical_text[:10] + label * 1000
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
    """Custom collate function."""
    batch_size = len(batch)
    labels = torch.stack([item['label'] for item in batch])
    
    # Handle WSI features
    wsi_list = []
    max_patches = 0
    for item in batch:
        if item['wsi_features'] is not None:
            wsi_list.append(item['wsi_features'])
            max_patches = max(max_patches, item['wsi_features'].shape[0])
        else:
            wsi_list.append(None)
    
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
    
    # Handle genomic features
    genomic_list = [item['genomic'] for item in batch if item['genomic'] is not None]
    if len(genomic_list) > 0:
        genomic = torch.zeros(batch_size, 2000)
        for i, item in enumerate(batch):
            if item['genomic'] is not None:
                genomic[i] = item['genomic']
    else:
        genomic = None
    
    # Handle clinical text
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

print("Creating datasets...")
train_dataset = SyntheticMultimodalDataset(num_samples=800, missing_rate=0.2)
val_dataset = SyntheticMultimodalDataset(num_samples=100, missing_rate=0.2)
test_dataset = SyntheticMultimodalDataset(num_samples=100, missing_rate=0.2)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Initialize model
print("\nInitializing model...")
model = MultimodalFusionModel(embed_dim=256).to(device)
classifier = ClassificationHead(input_dim=256, num_classes=4).to(device)

total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters())
print(f"Total parameters: {total_params:,}")

# Optimizer and loss
optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def train_epoch(model, classifier, dataloader, optimizer, criterion, device):
    model.train()
    classifier.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
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
        for batch in tqdm(dataloader, desc="Evaluating"):
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

# Training loop
print("\nStarting training...")
num_epochs = 10  # Reduced for faster demo
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

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
        }, 'models/best_model.pth')
        print(f"✓ Saved best model (val_acc: {val_acc:.4f})")

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

# Plot training curves
print("\nGenerating training curves...")
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
plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved training_curves.png")

# Test set evaluation
print("\nEvaluating on test set...")
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
classifier.load_state_dict(checkpoint['classifier_state_dict'])

test_loss, test_acc, test_preds, test_labels = evaluate(model, classifier, test_loader, criterion, device)

print(f"\nTest Results:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=[f'Class {i}' for i in range(4)]))

# Confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Class {i}' for i in range(4)],
            yticklabels=[f'Class {i}' for i in range(4)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")

# t-SNE visualization
print("\nGenerating t-SNE visualization...")
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

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=all_labels, cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Class')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of Multimodal Embeddings')
plt.grid(True, alpha=0.3)
plt.savefig('results/tsne_embeddings.png', dpi=300, bbox_inches='tight')
print("✓ Saved tsne_embeddings.png")

print("\n" + "="*60)
print("DEMO COMPLETE!")
print("="*60)
print(f"Final Results:")
print(f"  - Test Accuracy: {test_acc:.4f}")
print(f"  - Best Val Accuracy: {best_val_acc:.4f}")
print(f"\nGenerated files:")
print(f"  - results/training_curves.png")
print(f"  - results/confusion_matrix.png")
print(f"  - results/tsne_embeddings.png")
print(f"  - models/best_model.pth")
print("="*60)
