#!/usr/bin/env python3
"""PathologyFL Quick Start Example."""

import time
import random
from typing import Dict, List

class PathologyFLQuickStart:
    """Quick start example for PathologyFL."""
    
    def __init__(self):
        self.coordinator = None
        self.clients = []
        
    def setup_coordinator(self):
        """Set up the federated learning coordinator."""
        print("🏥 Setting up PathologyFL Coordinator...")
        
        self.coordinator = {
            "host": "localhost",
            "port": 8080,
            "hospitals": {},
            "global_model": {},
            "round": 0
        }
        
        print("✅ Coordinator ready at localhost:8080")
        
    def register_hospital(self, hospital_id: str, metadata: Dict):
        """Register a hospital with the coordinator."""
        print(f"📋 Registering hospital: {hospital_id}")
        
        # Calculate hospital weight based on expertise
        weight = 1.0
        if metadata.get("hospital_type") == "cancer_center":
            weight *= 2.0
        
        accuracy = metadata.get("diagnostic_accuracy", 0.8)
        weight *= accuracy
        
        self.coordinator["hospitals"][hospital_id] = {
            "metadata": metadata,
            "weight": weight,
            "last_update": time.time()
        }
        
        print(f"   Weight: {weight:.2f}")
        
    def create_client(self, hospital_id: str):
        """Create a federated learning client."""
        print(f"🏥 Creating client for {hospital_id}")
        
        client = {
            "hospital_id": hospital_id,
            "local_model": {
                "conv_layer": [random.uniform(-0.1, 0.1) for _ in range(100)],
                "fc_layer": [random.uniform(-0.1, 0.1) for _ in range(50)]
            },
            "training_data_size": random.randint(1000, 10000)
        }
        
        self.clients.append(client)
        print(f"   Training data: {client['training_data_size']} samples")
        
    def simulate_local_training(self, client: Dict):
        """Simulate local training at a hospital."""
        print(f"🔬 Training at {client['hospital_id']}...")
        
        # Simulate training by slightly modifying model parameters
        for layer in client["local_model"]:
            for i in range(len(client["local_model"][layer])):
                # Add small random update
                update = random.uniform(-0.01, 0.01)
                client["local_model"][layer][i] += update
        
        # Simulate training time
        time.sleep(0.1)
        
        print(f"   ✅ Local training complete")
        
    def aggregate_models(self):
        """Aggregate models from all clients."""
        print("🔄 Aggregating models from all hospitals...")
        
        if not self.clients:
            print("   ❌ No clients available")
            return
        
        # Initialize global model
        global_model = {}
        total_weight = 0.0
        
        # Weighted aggregation
        for client in self.clients:
            hospital_id = client["hospital_id"]
            weight = self.coordinator["hospitals"][hospital_id]["weight"]
            total_weight += weight
            
            for layer, params in client["local_model"].items():
                if layer not in global_model:
                    global_model[layer] = [0.0] * len(params)
                
                for i, param in enumerate(params):
                    global_model[layer][i] += param * weight
        
        # Normalize by total weight
        for layer in global_model:
            for i in range(len(global_model[layer])):
                global_model[layer][i] /= total_weight
        
        self.coordinator["global_model"] = global_model
        self.coordinator["round"] += 1
        
        print(f"   ✅ Global model updated (Round {self.coordinator['round']})")
        
    def evaluate_global_model(self):
        """Evaluate the global model performance."""
        print("📊 Evaluating global model...")
        
        # Simulate evaluation metrics
        accuracy = random.uniform(0.85, 0.95)
        sensitivity = random.uniform(0.80, 0.90)
        specificity = random.uniform(0.88, 0.95)
        auc = random.uniform(0.90, 0.98)
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Sensitivity: {sensitivity:.3f}")
        print(f"   Specificity: {specificity:.3f}")
        print(f"   AUC: {auc:.3f}")
        
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc
        }
        
    def run_federated_round(self):
        """Run one complete federated learning round."""
        print(f"\n🚀 Starting Federated Learning Round {self.coordinator['round'] + 1}")
        print("=" * 60)
        
        # Local training at each hospital
        for client in self.clients:
            self.simulate_local_training(client)
        
        # Aggregate models
        self.aggregate_models()
        
        # Evaluate global model
        metrics = self.evaluate_global_model()
        
        return metrics
        
    def run_complete_example(self):
        """Run complete PathologyFL example."""
        print("🏥 PathologyFL Quick Start Example")
        print("=" * 60)
        
        # Step 1: Setup coordinator
        self.setup_coordinator()
        
        # Step 2: Register hospitals
        hospitals = [
            ("mayo_clinic", {
                "hospital_type": "cancer_center",
                "annual_cases": 15000,
                "diagnostic_accuracy": 0.96,
                "years_experience": 25
            }),
            ("johns_hopkins", {
                "hospital_type": "teaching_hospital", 
                "annual_cases": 12000,
                "diagnostic_accuracy": 0.94,
                "years_experience": 20
            }),
            ("community_hospital", {
                "hospital_type": "community",
                "annual_cases": 3000,
                "diagnostic_accuracy": 0.88,
                "years_experience": 10
            })
        ]
        
        for hospital_id, metadata in hospitals:
            self.register_hospital(hospital_id, metadata)
            self.create_client(hospital_id)
        
        # Step 3: Run federated learning rounds
        num_rounds = 5
        all_metrics = []
        
        for round_num in range(num_rounds):
            metrics = self.run_federated_round()
            all_metrics.append(metrics)
        
        # Step 4: Show final results
        print(f"\n📈 Final Results after {num_rounds} rounds:")
        print("=" * 60)
        
        final_metrics = all_metrics[-1]
        for metric, value in final_metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")
        
        # Show improvement
        if len(all_metrics) > 1:
            initial_acc = all_metrics[0]["accuracy"]
            final_acc = all_metrics[-1]["accuracy"]
            improvement = final_acc - initial_acc
            
            print(f"\nAccuracy improvement: {improvement:+.3f}")
            
        print(f"\n🎉 PathologyFL training complete!")
        print(f"Hospitals participated: {len(self.clients)}")
        print(f"Federated rounds: {num_rounds}")
        
        return all_metrics

def main():
    """Run the PathologyFL quick start example."""
    quickstart = PathologyFLQuickStart()
    results = quickstart.run_complete_example()
    
    # Show summary
    print(f"\n📋 Training Summary:")
    print(f"   Rounds completed: {len(results)}")
    print(f"   Final accuracy: {results[-1]['accuracy']:.3f}")
    print(f"   Final AUC: {results[-1]['auc']:.3f}")

if __name__ == "__main__":
    main()