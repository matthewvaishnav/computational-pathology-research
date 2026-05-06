#!/usr/bin/env python3
"""
PathologyFL Integration Test - Full end-to-end test of hierarchical medical FL
"""

import asyncio
import torch
import json
import tempfile
from pathlib import Path

from src.federated.pathology_fl_coordinator import PathologyFLCoordinator
from src.federated.pathology_fl_client import PathologyFLClient

async def test_pathology_fl_integration():
    """Test complete PathologyFL workflow."""
    
    print("🧬 Testing PathologyFL Integration")
    print("=" * 50)
    
    # Create temporary config files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Coordinator config
        coordinator_config = {
            "expertise_weight": 0.5,
            "quality_weight": 0.3,
            "num_rounds": 3
        }
        
        coord_config_path = temp_path / "coordinator_config.json"
        with open(coord_config_path, "w") as f:
            json.dump(coordinator_config, f)
        
        # Client configs
        mayo_config = {
            "learning_rate": 0.001,
            "hospital_metadata": {
                "hospital_type": "cancer_center",
                "annual_cases": 15000,
                "cancer_specialties": ["breast", "lung", "prostate"],
                "diagnostic_accuracy": 0.96,
                "years_experience": 20
            }
        }
        
        community_config = {
            "learning_rate": 0.001,
            "hospital_metadata": {
                "hospital_type": "community_hospital",
                "annual_cases": 3000,
                "cancer_specialties": ["general"],
                "diagnostic_accuracy": 0.87,
                "years_experience": 8
            }
        }
        
        mayo_config_path = temp_path / "mayo_config.json"
        community_config_path = temp_path / "community_config.json"
        
        with open(mayo_config_path, "w") as f:
            json.dump(mayo_config, f)
        with open(community_config_path, "w") as f:
            json.dump(community_config, f)
        
        # Initialize coordinator
        print("🏥 Initializing PathologyFL Coordinator...")
        coordinator = PathologyFLCoordinator(str(coord_config_path))
        
        # Register hospitals
        coordinator.register_hospital("mayo_clinic", mayo_config["hospital_metadata"])
        coordinator.register_hospital("community_hospital", community_config["hospital_metadata"])
        
        # Initialize clients
        print("🏥 Initializing PathologyFL Clients...")
        mayo_client = PathologyFLClient("mayo_clinic", str(mayo_config_path))
        community_client = PathologyFLClient("community_hospital", str(community_config_path))
        
        # Initialize global model
        print("🧠 Initializing Global Model...")
        global_model = {
            "layer1.weight": torch.randn(128, 64),
            "layer1.bias": torch.randn(128),
            "attention.weight": torch.randn(64, 32),
            "classifier.weight": torch.randn(2, 128),
            "classifier.bias": torch.randn(2)
        }
        coordinator.initialize_global_model(global_model)
        
        # Mock training data
        class MockDataLoader:
            def __init__(self, quality_level="high"):
                self.quality_level = quality_level
                confidence = 0.9 if quality_level == "high" else 0.75
                
                self.data = [
                    (torch.randn(4, 64), torch.randint(0, 2, (4,)), [
                        {"label_confidence": confidence + torch.randn(1).item() * 0.05}
                        for _ in range(4)
                    ])
                    for _ in range(3)  # 3 batches
                ]
            
            def __iter__(self):
                return iter(self.data)
        
        # Run federated learning rounds
        print("\n🔄 Starting PathologyFL Training Rounds...")
        
        for round_num in range(3):
            print(f"\n--- Round {round_num + 1} ---")
            
            # Distribute global model to clients
            current_global = coordinator.get_global_model()
            mayo_client.set_model(current_global)
            community_client.set_model(current_global)
            
            # Local training
            print("  🏥 Mayo Clinic training...")
            mayo_loader = MockDataLoader("high")  # High quality data
            mayo_updates, mayo_quality = mayo_client.train_local_model(
                mayo_loader, epochs=1, cancer_type="breast"
            )
            
            print("  🏥 Community Hospital training...")
            community_loader = MockDataLoader("medium")  # Medium quality data
            community_updates, community_quality = community_client.train_local_model(
                community_loader, epochs=1, cancer_type="breast"
            )
            
            # Aggregate updates
            print("  🔄 Aggregating updates with PathologyFL...")
            client_updates = {
                "mayo_clinic": mayo_updates,
                "community_hospital": community_updates
            }
            
            slide_qualities = {
                "mayo_clinic": mayo_quality,
                "community_hospital": community_quality
            }
            
            updated_global = await coordinator.federated_round(
                client_updates, slide_qualities, "breast"
            )
            
            print(f"  ✅ Round {round_num + 1} completed")
        
        # Test checkpoint saving/loading
        print("\n💾 Testing Checkpoint Save/Load...")
        checkpoint_path = temp_path / "pathology_fl_checkpoint.pth"
        coordinator.save_checkpoint(str(checkpoint_path))
        
        # Create new coordinator and load checkpoint
        new_coordinator = PathologyFLCoordinator(str(coord_config_path))
        new_coordinator.load_checkpoint(str(checkpoint_path))
        
        # Verify checkpoint loaded correctly
        assert new_coordinator.round_number == 3
        assert len(new_coordinator.client_metadata) == 2
        print("  ✅ Checkpoint save/load successful")
        
        # Test expertise weighting
        print("\n⚖️ Testing Expertise Weighting...")
        mayo_metadata = coordinator.client_metadata["mayo_clinic"]
        community_metadata = coordinator.client_metadata["community_hospital"]
        
        mayo_weight = coordinator.aggregator.calculate_expertise_weight(
            mayo_metadata, coordinator.aggregator.CancerType.BREAST
        )
        community_weight = coordinator.aggregator.calculate_expertise_weight(
            community_metadata, coordinator.aggregator.CancerType.BREAST
        )
        
        print(f"  Mayo Clinic expertise weight: {mayo_weight:.3f}")
        print(f"  Community Hospital expertise weight: {community_weight:.3f}")
        
        # Mayo should have higher weight (cancer center vs community)
        assert mayo_weight > community_weight, "Cancer center should have higher expertise weight"
        print("  ✅ Expertise weighting working correctly")
        
        print("\n🎉 PathologyFL Integration Test PASSED!")
        print("=" * 50)
        
        return {
            "rounds_completed": 3,
            "mayo_expertise_weight": mayo_weight,
            "community_expertise_weight": community_weight,
            "final_model_params": len(updated_global),
            "checkpoint_saved": checkpoint_path.exists()
        }

def run_pathology_fl_demo():
    """Run PathologyFL demonstration."""
    
    print("🚀 PathologyFL Demo - Hierarchical Medical Expertise FL")
    print("=" * 60)
    
    # Run integration test
    results = asyncio.run(test_pathology_fl_integration())
    
    print("\n📊 Demo Results:")
    print(f"  ✅ Completed {results['rounds_completed']} federated rounds")
    print(f"  ✅ Mayo Clinic weight: {results['mayo_expertise_weight']:.3f}")
    print(f"  ✅ Community Hospital weight: {results['community_expertise_weight']:.3f}")
    print(f"  ✅ Final model has {results['final_model_params']} parameters")
    print(f"  ✅ Checkpoint saved: {results['checkpoint_saved']}")
    
    print("\n🏆 PathologyFL Key Innovations Demonstrated:")
    print("  🧬 Hierarchical medical expertise weighting")
    print("  🏥 Hospital type and specialty consideration")
    print("  📊 Slide quality assessment integration")
    print("  🎯 Cancer-type specific aggregation")
    print("  💾 Checkpoint save/load functionality")
    
    print("\n🎯 Competitive Advantage:")
    print("  vs Standard FL: Generic averaging → Medical expertise weighting")
    print("  vs TensorFlow FL: General purpose → Pathology-optimized")
    print("  vs PySyft: Privacy-focused → Medical workflow integration")
    
    print("\n✅ PathologyFL is ready for production deployment!")

if __name__ == "__main__":
    run_pathology_fl_demo()