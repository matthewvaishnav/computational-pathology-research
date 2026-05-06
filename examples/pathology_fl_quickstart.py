#!/usr/bin/env python3
"""
PathologyFL Quick Start Example
"""

import asyncio
from src.federated.pathology_fl_coordinator import PathologyFLCoordinator
from src.federated.pathology_fl_client import PathologyFLClient

async def pathology_fl_quickstart():
    """Quick start example for PathologyFL."""
    
    print("🧬 PathologyFL Quick Start")
    print("=" * 30)
    
    # Initialize coordinator
    coordinator = PathologyFLCoordinator("configs/pathology_fl_config.yaml")
    
    # Register hospitals
    coordinator.register_hospital("mayo_clinic", {
        "hospital_type": "cancer_center",
        "annual_cases": 15000,
        "cancer_specialties": ["breast", "lung"],
        "diagnostic_accuracy": 0.96,
        "years_experience": 20
    })
    
    print("✅ PathologyFL coordinator initialized")
    print("✅ Hospital registered with medical expertise")
    print("🚀 Ready for federated learning!")

if __name__ == "__main__":
    asyncio.run(pathology_fl_quickstart())