"""
Demo: Hospital Client for Federated Learning

Demonstrates how a hospital can participate in federated learning
while keeping patient data local and private.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client.hospital_client import HospitalClient


class SimplePathologyModel(nn.Module):
    """Simple pathology model for demonstration."""
    
    def __init__(self, input_dim=128, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def create_synthetic_hospital_data(num_samples=500, input_dim=128):
    """
    Create synthetic hospital data for demonstration.
    
    In production, this would be real patient data that never leaves the hospital.
    """
    # Generate synthetic features (e.g., patch embeddings from WSI)
    X = torch.randn(num_samples, input_dim)
    
    # Generate synthetic labels (e.g., tumor vs normal)
    y = torch.randint(0, 2, (num_samples,))
    
    # Create dataset and loaders
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def demo_basic_usage():
    """Demo: Basic hospital client usage."""
    print("=" * 60)
    print("Demo 1: Basic Hospital Client Usage")
    print("=" * 60)
    
    # Initialize model
    model = SimplePathologyModel(input_dim=128, num_classes=2)
    
    # Create hospital client WITHOUT privacy (for demo)
    client = HospitalClient(
        hospital_id="hospital_demo_001",
        model=model,
        coordinator_host="localhost",
        coordinator_port=50051,
        use_privacy=False,  # Disabled for demo
    )
    
    print(f"\n✓ Hospital client initialized: {client.hospital_id}")
    
    # Load local hospital data
    train_loader, val_loader = create_synthetic_hospital_data(num_samples=500)
    client.load_local_data(train_loader, val_loader)
    
    print(f"✓ Loaded {client.local_data_size} local training samples")
    
    # Get client info
    info = client.get_client_info()
    print(f"\nClient Information:")
    print(f"  - Hospital ID: {info['hospital_id']}")
    print(f"  - Local data size: {info['local_data_size']}")
    print(f"  - Model parameters: {info['total_parameters']:,}")
    print(f"  - Model size: {info['model_size_mb']:.2f} MB")
    print(f"  - Privacy enabled: {info['privacy_enabled']}")
    
    # Train locally (without coordinator for demo)
    print(f"\n✓ Training locally...")
    metrics = client.train_local_model(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.001,
    )
    
    print(f"  - Training loss: {metrics['loss']:.4f}")
    print(f"  - Training accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Training time: {metrics['training_time']:.2f}s")
    
    # Evaluate
    print(f"\n✓ Evaluating on validation data...")
    eval_metrics = client.evaluate_local_model()
    
    print(f"  - Validation loss: {eval_metrics['test_loss']:.4f}")
    print(f"  - Validation accuracy: {eval_metrics['test_accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo 1 Complete!")
    print("=" * 60)


def demo_privacy_enabled():
    """Demo: Hospital client with differential privacy."""
    print("\n" + "=" * 60)
    print("Demo 2: Hospital Client with Differential Privacy")
    print("=" * 60)
    
    # Initialize model
    model = SimplePathologyModel(input_dim=128, num_classes=2)
    
    # Create hospital client WITH privacy
    client = HospitalClient(
        hospital_id="hospital_demo_002",
        model=model,
        coordinator_host="localhost",
        coordinator_port=50051,
        use_privacy=True,  # Enable privacy
        privacy_epsilon=1.0,  # Privacy budget
        privacy_delta=1e-5,
        max_grad_norm=1.0,
    )
    
    print(f"\n✓ Hospital client initialized with privacy protection")
    
    # Load local hospital data
    train_loader, val_loader = create_synthetic_hospital_data(num_samples=500)
    client.load_local_data(train_loader, val_loader)
    
    print(f"✓ Loaded {client.local_data_size} local training samples")
    
    # Get privacy budget status
    privacy_status = client.get_privacy_budget_status()
    print(f"\nPrivacy Configuration:")
    print(f"  - Privacy enabled: {privacy_status['privacy_enabled']}")
    print(f"  - Epsilon budget: {privacy_status['epsilon_budget']}")
    print(f"  - Epsilon used: {privacy_status['epsilon_used']:.4f}")
    print(f"  - Epsilon remaining: {privacy_status['epsilon_remaining']:.4f}")
    print(f"  - Delta: {privacy_status['delta']:.2e}")
    print(f"  - Max gradient norm: {privacy_status['max_grad_norm']}")
    
    # Train with privacy
    print(f"\n✓ Training with differential privacy...")
    metrics = client.train_local_model(
        num_epochs=2,
        batch_size=32,
        learning_rate=0.001,
    )
    
    print(f"  - Training loss: {metrics['loss']:.4f}")
    print(f"  - Training accuracy: {metrics['accuracy']:.4f}")
    
    # Check privacy budget after training
    privacy_status = client.get_privacy_budget_status()
    print(f"\nPrivacy Budget After Training:")
    print(f"  - Epsilon used: {privacy_status['epsilon_used']:.4f}")
    print(f"  - Epsilon remaining: {privacy_status['epsilon_remaining']:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo 2 Complete!")
    print("=" * 60)


def demo_model_updates():
    """Demo: Computing and sending model updates."""
    print("\n" + "=" * 60)
    print("Demo 3: Model Updates (Data Privacy)")
    print("=" * 60)
    
    # Initialize model
    model = SimplePathologyModel(input_dim=128, num_classes=2)
    
    # Create hospital client
    client = HospitalClient(
        hospital_id="hospital_demo_003",
        model=model,
        use_privacy=False,
    )
    
    # Load local data
    train_loader, val_loader = create_synthetic_hospital_data(num_samples=500)
    client.load_local_data(train_loader, val_loader)
    
    print(f"\n✓ Hospital client initialized")
    print(f"✓ Loaded {client.local_data_size} local samples")
    
    # Store initial model state
    initial_state = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }
    
    print(f"\n✓ Stored initial model state")
    
    # Train locally
    print(f"✓ Training locally...")
    metrics = client.train_local_model(num_epochs=2, batch_size=32, learning_rate=0.001)
    
    # Compute model update
    print(f"✓ Computing model update...")
    model_update = client.compute_model_update(initial_state)
    
    print(f"\nModel Update Statistics:")
    print(f"  - Number of parameters: {len(model_update)}")
    
    # Calculate update magnitude
    total_update_norm = sum(
        torch.norm(update).item() for update in model_update.values()
    )
    print(f"  - Total update magnitude: {total_update_norm:.4f}")
    
    # Show that only updates are sent, not raw data
    print(f"\n✓ Key Point: Only model updates are computed")
    print(f"  - Raw patient data: NEVER leaves hospital")
    print(f"  - Model updates: Sent to coordinator")
    print(f"  - Data privacy: PRESERVED")
    
    print("\n" + "=" * 60)
    print("Demo 3 Complete!")
    print("=" * 60)


def demo_federated_round_simulation():
    """Demo: Simulating a federated learning round."""
    print("\n" + "=" * 60)
    print("Demo 4: Federated Learning Round Simulation")
    print("=" * 60)
    
    # Initialize model
    model = SimplePathologyModel(input_dim=128, num_classes=2)
    
    # Create hospital client
    client = HospitalClient(
        hospital_id="hospital_demo_004",
        model=model,
        use_privacy=True,
        privacy_epsilon=2.0,
    )
    
    # Load local data
    train_loader, val_loader = create_synthetic_hospital_data(num_samples=500)
    client.load_local_data(train_loader, val_loader)
    
    print(f"\n✓ Hospital client ready for federated learning")
    print(f"✓ Local data: {client.local_data_size} samples")
    
    # Simulate federated learning rounds
    num_rounds = 3
    print(f"\n✓ Simulating {num_rounds} federated learning rounds...")
    
    for round_id in range(1, num_rounds + 1):
        print(f"\n--- Round {round_id} ---")
        
        # Store initial state (simulating global model)
        initial_state = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # Local training
        print(f"  Training locally...")
        metrics = client.train_local_model(
            num_epochs=1,
            batch_size=32,
            learning_rate=0.001,
        )
        
        print(f"  - Loss: {metrics['loss']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        
        # Compute update
        model_update = client.compute_model_update(initial_state)
        
        # Privacy status
        privacy_status = client.get_privacy_budget_status()
        print(f"  - Privacy epsilon used: {privacy_status['epsilon_used']:.4f}")
        print(f"  - Privacy epsilon remaining: {privacy_status['epsilon_remaining']:.4f}")
        
        # In real scenario, update would be sent to coordinator here
        print(f"  ✓ Model update computed (would be sent to coordinator)")
    
    print(f"\n✓ Federated learning simulation complete!")
    print(f"✓ Patient data remained local throughout all rounds")
    
    print("\n" + "=" * 60)
    print("Demo 4 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Hospital Client for Federated Learning - Demo")
    print("=" * 60)
    print("\nThis demo shows how hospitals can participate in")
    print("federated learning while keeping patient data local.")
    print("\n" + "=" * 60)
    
    # Run demos
    demo_basic_usage()
    demo_privacy_enabled()
    demo_model_updates()
    demo_federated_round_simulation()
    
    print("\n" + "=" * 60)
    print("All Demos Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Patient data NEVER leaves the hospital")
    print("  2. Only model updates are sent to coordinator")
    print("  3. Differential privacy protects individual patients")
    print("  4. Hospitals maintain full control over their data")
    print("=" * 60)
