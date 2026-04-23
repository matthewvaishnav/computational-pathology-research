#!/usr/bin/env python3
"""
PACS Integration System Example

This example demonstrates how to use the HistoCore PACS Integration System
for clinical deployment scenarios.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clinical.pacs import (
    PACSAdapter, WorkflowOrchestrator, ConfigurationManager,
    StudyInfo, AnalysisResults, DetectedRegion, DiagnosticRecommendation
)
from src.clinical.workflow import ClinicalWorkflowSystem, ClinicalWorkflowConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting PACS Integration System Example")
    
    # Example 1: Basic PACS Operations
    basic_pacs_operations()
    
    # Example 2: Automated Workflow
    automated_workflow_example()
    
    # Example 3: Configuration Management
    configuration_management_example()
    
    logger.info("PACS Integration System Example Complete")


def basic_pacs_operations():
    """Demonstrate basic PACS operations."""
    logger.info("=== Basic PACS Operations Example ===")
    
    try:
        # Initialize PACS adapter with development configuration
        with PACSAdapter(config_profile="development") as pacs:
            
            # Test PACS connection
            logger.info("Testing PACS connection...")
            connection_result = pacs.test_connection()
            
            if connection_result.success:
                logger.info("✓ PACS connection successful")
            else:
                logger.warning(f"✗ PACS connection failed: {connection_result.message}")
                return
            
            # Query for studies
            logger.info("Querying for WSI studies...")
            studies, query_result = pacs.query_studies(
                modality="SM",  # Slide Microscopy
                max_results=10
            )
            
            if query_result.success:
                logger.info(f"✓ Found {len(studies)} studies")
                
                # Display study information
                for i, study in enumerate(studies[:3]):  # Show first 3
                    logger.info(f"  Study {i+1}: {study.study_instance_uid}")
                    logger.info(f"    Patient: {study.patient_name} ({study.patient_id})")
                    logger.info(f"    Date: {study.study_date}")
                    logger.info(f"    Description: {study.study_description}")
                
                # Retrieve a study (if available)
                if studies:
                    study_to_retrieve = studies[0]
                    logger.info(f"Retrieving study: {study_to_retrieve.study_instance_uid}")
                    
                    retrieval_result = pacs.retrieve_study(
                        study_instance_uid=study_to_retrieve.study_instance_uid,
                        destination_path="./data/retrieved_example"
                    )
                    
                    if retrieval_result.success:
                        logger.info("✓ Study retrieval successful")
                        retrieved_files = retrieval_result.data.get("retrieved_files", [])
                        logger.info(f"  Retrieved {len(retrieved_files)} files")
                    else:
                        logger.warning(f"✗ Study retrieval failed: {retrieval_result.message}")
                
                # Store analysis results example
                logger.info("Storing example analysis results...")
                
                # Create mock analysis results
                analysis_results = create_mock_analysis_results(studies[0] if studies else None)
                
                if analysis_results:
                    storage_result = pacs.store_analysis_results(
                        analysis_results=analysis_results,
                        original_study_uid=analysis_results.study_instance_uid
                    )
                    
                    if storage_result.success:
                        logger.info("✓ Analysis results stored successfully")
                        sop_uid = storage_result.data.get("sop_instance_uid")
                        logger.info(f"  Structured Report SOP UID: {sop_uid}")
                    else:
                        logger.warning(f"✗ Storage failed: {storage_result.message}")
            
            else:
                logger.warning(f"✗ Query failed: {query_result.message}")
            
            # Get endpoint status
            logger.info("Checking endpoint status...")
            status = pacs.get_endpoint_status()
            
            for endpoint_id, endpoint_status in status.items():
                connection_status = endpoint_status["connection_status"]
                logger.info(f"  {endpoint_id}: {connection_status}")
    
    except Exception as e:
        logger.error(f"Basic PACS operations failed: {str(e)}")


def automated_workflow_example():
    """Demonstrate automated workflow orchestration."""
    logger.info("=== Automated Workflow Example ===")
    
    try:
        # Initialize PACS adapter
        pacs = PACSAdapter(config_profile="development")
        
        # Initialize clinical workflow system
        clinical_config = ClinicalWorkflowConfig(
            taxonomy_config="configs/clinical/taxonomy.yaml",
            enable_dicom=True,
            enable_audit_logging=True
        )
        
        # Note: This would require the actual clinical workflow system to be set up
        # For this example, we'll create a mock
        logger.info("Note: Clinical workflow system integration requires full HistoCore setup")
        
        # Create workflow orchestrator
        # clinical_workflow = ClinicalWorkflowSystem(clinical_config)
        # orchestrator = WorkflowOrchestrator(
        #     pacs_adapter=pacs,
        #     clinical_workflow=clinical_workflow,
        #     poll_interval=timedelta(minutes=1),  # Fast polling for demo
        #     max_concurrent_studies=3
        # )
        
        logger.info("Workflow orchestrator would:")
        logger.info("  1. Poll PACS every minute for new WSI studies")
        logger.info("  2. Automatically retrieve and process studies")
        logger.info("  3. Run AI analysis through clinical workflow")
        logger.info("  4. Store results back to PACS as Structured Reports")
        logger.info("  5. Handle errors and retry failed operations")
        
        # Start automated processing
        # logger.info("Starting automated workflow...")
        # orchestrator.start_automated_polling()
        
        # Let it run for a short time
        # import time
        # time.sleep(30)  # Run for 30 seconds
        
        # Get processing status
        # status = orchestrator.get_processing_status()
        # logger.info(f"Processing status: {status}")
        
        # Stop automated processing
        # orchestrator.stop_automated_polling()
        
        logger.info("✓ Automated workflow example complete")
    
    except Exception as e:
        logger.error(f"Automated workflow example failed: {str(e)}")


def configuration_management_example():
    """Demonstrate configuration management capabilities."""
    logger.info("=== Configuration Management Example ===")
    
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager("configs/pacs")
        
        # List available profiles
        profiles = config_manager.list_available_profiles()
        logger.info(f"Available configuration profiles: {profiles}")
        
        # Load development configuration
        dev_config = config_manager.load_configuration("development")
        logger.info(f"Loaded development config: {dev_config.profile_name}")
        logger.info(f"  Endpoints: {len(dev_config.pacs_endpoints)}")
        
        for endpoint_id, endpoint in dev_config.pacs_endpoints.items():
            logger.info(f"    {endpoint_id}: {endpoint.host}:{endpoint.port} ({endpoint.vendor.value})")
        
        # Validate configuration
        validation = config_manager.validate_configuration(dev_config)
        
        if validation.is_valid:
            logger.info("✓ Configuration validation passed")
        else:
            logger.warning("✗ Configuration validation failed:")
            for error in validation.errors:
                logger.warning(f"    {error}")
        
        # Create default configuration
        logger.info("Creating default configuration...")
        default_config = config_manager.create_default_configuration()
        
        # Save configuration example
        save_result = config_manager.save_configuration(
            config=default_config,
            profile="example_generated",
            encrypted=False
        )
        
        if save_result.success:
            logger.info("✓ Configuration saved successfully")
        else:
            logger.warning(f"✗ Configuration save failed: {save_result.message}")
        
        # Update endpoint settings example
        update_result = config_manager.update_endpoint_settings(
            profile="development",
            endpoint_id="local_test",
            settings={
                "description": "Updated description from example",
                "performance_config": {
                    "max_concurrent_studies": 5
                }
            }
        )
        
        if update_result.success:
            logger.info("✓ Endpoint settings updated successfully")
        else:
            logger.warning(f"✗ Endpoint update failed: {update_result.message}")
        
        logger.info("✓ Configuration management example complete")
    
    except Exception as e:
        logger.error(f"Configuration management example failed: {str(e)}")


def create_mock_analysis_results(study: StudyInfo = None) -> AnalysisResults:
    """Create mock analysis results for demonstration."""
    if not study:
        # Create a mock study
        study_uid = "1.2.3.4.5.6.7.8.9.10"
        series_uid = "1.2.3.4.5.6.7.8.9.11"
    else:
        study_uid = study.study_instance_uid
        series_uid = study.study_instance_uid  # Simplified
    
    # Create detected regions
    detected_regions = [
        DetectedRegion(
            region_id="region_001",
            coordinates=(150, 200, 100, 80),
            confidence=0.92,
            region_type="malignant_tissue",
            description="High confidence malignant region detected"
        ),
        DetectedRegion(
            region_id="region_002", 
            coordinates=(300, 150, 60, 60),
            confidence=0.78,
            region_type="suspicious_area",
            description="Suspicious area requiring further review"
        )
    ]
    
    # Create diagnostic recommendations
    recommendations = [
        DiagnosticRecommendation(
            recommendation_id="rec_001",
            recommendation_text="Recommend immediate pathologist review due to high malignancy confidence",
            confidence=0.95,
            urgency_level="HIGH",
            supporting_evidence=["High confidence malignant region", "Morphological features consistent with carcinoma"]
        ),
        DiagnosticRecommendation(
            recommendation_id="rec_002",
            recommendation_text="Consider additional immunohistochemical staining for confirmation",
            confidence=0.85,
            urgency_level="MEDIUM",
            supporting_evidence=["Suspicious morphology", "Differential diagnosis required"]
        )
    ]
    
    return AnalysisResults(
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        algorithm_name="HistoCore AI Pathology Classifier",
        algorithm_version="2.1.0",
        confidence_score=0.89,
        detected_regions=detected_regions,
        diagnostic_recommendations=recommendations,
        processing_timestamp=datetime.now(),
        primary_diagnosis="Invasive Ductal Carcinoma",
        probability_distribution={
            "Invasive Ductal Carcinoma": 0.89,
            "Invasive Lobular Carcinoma": 0.07,
            "Benign": 0.04
        }
    )


if __name__ == "__main__":
    main()