"""
Site Preparation Module for Medical AI Revolution
Handles technical requirements assessment and deployment planning for pilot hospitals.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class TechnicalRequirements:
    """Technical requirements for hospital deployment."""
    min_cpu_cores: int = 16
    min_ram_gb: int = 64
    min_gpu_memory_gb: int = 16
    min_storage_tb: int = 10
    network_bandwidth_mbps: int = 1000
    os_requirements: List[str] = None
    software_dependencies: List[str] = None
    
    def __post_init__(self):
        if self.os_requirements is None:
            self.os_requirements = ["Ubuntu 20.04+", "CentOS 8+", "RHEL 8+"]
        if self.software_dependencies is None:
            self.software_dependencies = [
                "Docker 20.10+",
                "NVIDIA Driver 470+",
                "CUDA 11.8+",
                "Python 3.9+",
                "PostgreSQL 13+"
            ]

@dataclass
class HospitalSite:
    """Hospital site configuration."""
    site_id: str
    name: str
    location: str
    contact_email: str
    it_contact: str
    pathology_volume: int  # slides per month
    current_systems: List[str]
    integration_requirements: List[str]
    compliance_requirements: List[str]
    
class SitePreparationManager:
    """Manages site preparation for pilot hospital deployments."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/deployment/site_preparation.yaml"
        self.sites: Dict[str, HospitalSite] = {}
        self.requirements = TechnicalRequirements()
        
    def assess_technical_requirements(self, site_id: str) -> Dict[str, Any]:
        """Assess technical requirements for a hospital site."""
        logger.info(f"Assessing technical requirements for site: {site_id}")
        
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
            
        site = self.sites[site_id]
        
        # Calculate requirements based on pathology volume
        volume_multiplier = max(1.0, site.pathology_volume / 1000)
        
        assessment = {
            "site_id": site_id,
            "site_name": site.name,
            "recommended_specs": {
                "cpu_cores": int(self.requirements.min_cpu_cores * volume_multiplier),
                "ram_gb": int(self.requirements.min_ram_gb * volume_multiplier),
                "gpu_memory_gb": max(16, int(self.requirements.min_gpu_memory_gb * volume_multiplier)),
                "storage_tb": int(self.requirements.min_storage_tb * volume_multiplier),
                "network_bandwidth_mbps": self.requirements.network_bandwidth_mbps
            },
            "software_requirements": self.requirements.software_dependencies.copy(),
            "os_requirements": self.requirements.os_requirements.copy(),
            "integration_points": site.integration_requirements,
            "compliance_needs": site.compliance_requirements,
            "estimated_setup_time_days": self._estimate_setup_time(site),
            "risk_factors": self._identify_risk_factors(site)
        }
        
        return assessment
        
    def create_integration_plan(self, site_id: str) -> Dict[str, Any]:
        """Create integration plan for hospital systems."""
        logger.info(f"Creating integration plan for site: {site_id}")
        
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
            
        site = self.sites[site_id]
        
        plan = {
            "site_id": site_id,
            "integration_phases": [
                {
                    "phase": "Pre-deployment",
                    "duration_days": 5,
                    "tasks": [
                        "Network connectivity testing",
                        "Security assessment",
                        "Firewall configuration",
                        "VPN setup if required"
                    ]
                },
                {
                    "phase": "System Installation", 
                    "duration_days": 3,
                    "tasks": [
                        "Hardware verification",
                        "OS installation and configuration",
                        "Docker environment setup",
                        "GPU driver installation"
                    ]
                },
                {
                    "phase": "Application Deployment",
                    "duration_days": 2,
                    "tasks": [
                        "Medical AI system installation",
                        "Database setup and migration",
                        "Configuration management",
                        "Initial data loading"
                    ]
                },
                {
                    "phase": "Integration Testing",
                    "duration_days": 5,
                    "tasks": [
                        "LIS integration testing",
                        "EMR integration testing", 
                        "Scanner connectivity testing",
                        "End-to-end workflow testing"
                    ]
                }
            ],
            "integration_points": self._map_integration_points(site),
            "testing_protocols": self._generate_testing_protocols(site),
            "rollback_procedures": self._create_rollback_procedures(site)
        }
        
        return plan
        
    def generate_training_program(self, site_id: str) -> Dict[str, Any]:
        """Generate staff training program for hospital site."""
        logger.info(f"Generating training program for site: {site_id}")
        
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
            
        site = self.sites[site_id]
        
        program = {
            "site_id": site_id,
            "training_modules": [
                {
                    "module": "System Overview",
                    "duration_hours": 2,
                    "audience": ["Pathologists", "Lab Technicians", "IT Staff"],
                    "topics": [
                        "Medical AI system capabilities",
                        "Workflow integration",
                        "User interface overview",
                        "Safety and limitations"
                    ]
                },
                {
                    "module": "Pathologist Training",
                    "duration_hours": 8,
                    "audience": ["Pathologists"],
                    "topics": [
                        "AI-assisted diagnosis workflow",
                        "Interpretation of AI results",
                        "Quality assurance procedures",
                        "Case review and validation"
                    ]
                },
                {
                    "module": "Technical Training",
                    "duration_hours": 6,
                    "audience": ["Lab Technicians", "IT Staff"],
                    "topics": [
                        "System operation and monitoring",
                        "Troubleshooting procedures",
                        "Data management",
                        "Backup and recovery"
                    ]
                },
                {
                    "module": "Compliance Training",
                    "duration_hours": 4,
                    "audience": ["All Users"],
                    "topics": [
                        "HIPAA compliance",
                        "Data privacy requirements",
                        "Audit procedures",
                        "Incident reporting"
                    ]
                }
            ],
            "certification_requirements": [
                "Complete all training modules",
                "Pass competency assessment",
                "Supervised practice period (40 hours)",
                "Final certification exam"
            ],
            "ongoing_education": [
                "Monthly system updates training",
                "Quarterly best practices review",
                "Annual recertification",
                "Continuous quality improvement"
            ]
        }
        
        return program
        
    def prepare_go_live_checklist(self, site_id: str) -> Dict[str, Any]:
        """Prepare go-live checklist for hospital deployment."""
        logger.info(f"Preparing go-live checklist for site: {site_id}")
        
        if site_id not in self.sites:
            raise ValueError(f"Site {site_id} not found")
            
        checklist = {
            "site_id": site_id,
            "pre_go_live": [
                {
                    "category": "Technical Readiness",
                    "items": [
                        "All systems installed and configured",
                        "Integration testing completed successfully",
                        "Performance benchmarks met",
                        "Security assessment passed",
                        "Backup systems verified",
                        "Monitoring systems active"
                    ]
                },
                {
                    "category": "Staff Readiness", 
                    "items": [
                        "All staff completed training",
                        "Competency assessments passed",
                        "Super users identified and trained",
                        "Support procedures documented",
                        "Escalation paths established"
                    ]
                },
                {
                    "category": "Process Readiness",
                    "items": [
                        "Workflows documented and approved",
                        "Quality assurance procedures in place",
                        "Incident response procedures defined",
                        "Change management process established",
                        "Communication plan activated"
                    ]
                }
            ],
            "go_live_day": [
                "System status verification",
                "Staff availability confirmation", 
                "Support team on standby",
                "Communication channels open",
                "Monitoring dashboards active",
                "Rollback procedures ready"
            ],
            "post_go_live": [
                "24/7 monitoring for first week",
                "Daily check-ins with site staff",
                "Performance metrics collection",
                "Issue tracking and resolution",
                "User feedback collection",
                "Optimization recommendations"
            ],
            "success_criteria": [
                "System uptime > 99.5%",
                "Processing time < 30 seconds per slide",
                "User satisfaction > 85%",
                "Zero critical incidents",
                "All workflows functioning correctly"
            ]
        }
        
        return checklist
        
    def add_hospital_site(self, site: HospitalSite):
        """Add a hospital site to the preparation system."""
        self.sites[site.site_id] = site
        logger.info(f"Added hospital site: {site.name} ({site.site_id})")
        
    def _estimate_setup_time(self, site: HospitalSite) -> int:
        """Estimate setup time in days based on site complexity."""
        base_time = 15  # Base setup time in days
        
        # Add time based on integration complexity
        integration_complexity = len(site.integration_requirements)
        complexity_time = integration_complexity * 2
        
        # Add time based on compliance requirements
        compliance_complexity = len(site.compliance_requirements)
        compliance_time = compliance_complexity * 1
        
        # Add time based on pathology volume
        volume_time = max(0, (site.pathology_volume - 1000) // 500)
        
        total_time = base_time + complexity_time + compliance_time + volume_time
        return min(total_time, 45)  # Cap at 45 days
        
    def _identify_risk_factors(self, site: HospitalSite) -> List[str]:
        """Identify potential risk factors for deployment."""
        risks = []
        
        if site.pathology_volume > 5000:
            risks.append("High volume site - performance testing critical")
            
        if "Epic" in site.current_systems:
            risks.append("Epic integration complexity")
            
        if "HIPAA" in site.compliance_requirements:
            risks.append("Strict HIPAA compliance requirements")
            
        if len(site.integration_requirements) > 5:
            risks.append("Multiple system integrations required")
            
        return risks
        
    def _map_integration_points(self, site: HospitalSite) -> List[Dict[str, str]]:
        """Map integration points for the site."""
        integrations = []
        
        for system in site.current_systems:
            if "LIS" in system:
                integrations.append({
                    "system": system,
                    "type": "LIS",
                    "protocol": "HL7 v2.5",
                    "data_flow": "Bidirectional"
                })
            elif "EMR" in system or "Epic" in system:
                integrations.append({
                    "system": system,
                    "type": "EMR", 
                    "protocol": "FHIR R4",
                    "data_flow": "Outbound"
                })
            elif "Scanner" in system:
                integrations.append({
                    "system": system,
                    "type": "Scanner",
                    "protocol": "DICOM",
                    "data_flow": "Inbound"
                })
                
        return integrations
        
    def _generate_testing_protocols(self, site: HospitalSite) -> List[Dict[str, Any]]:
        """Generate testing protocols for the site."""
        protocols = [
            {
                "test_name": "System Performance Test",
                "duration_hours": 4,
                "test_cases": [
                    "Process 100 slides in parallel",
                    "Measure response times",
                    "Monitor resource utilization",
                    "Verify accuracy metrics"
                ]
            },
            {
                "test_name": "Integration Test",
                "duration_hours": 8,
                "test_cases": [
                    "End-to-end workflow testing",
                    "Data synchronization verification",
                    "Error handling validation",
                    "Failover testing"
                ]
            },
            {
                "test_name": "User Acceptance Test",
                "duration_hours": 16,
                "test_cases": [
                    "Pathologist workflow validation",
                    "Technician workflow validation",
                    "Report generation testing",
                    "User interface testing"
                ]
            }
        ]
        
        return protocols
        
    def _create_rollback_procedures(self, site: HospitalSite) -> Dict[str, List[str]]:
        """Create rollback procedures for the site."""
        procedures = {
            "immediate_rollback": [
                "Stop AI processing services",
                "Redirect traffic to backup systems",
                "Notify support team",
                "Document incident details"
            ],
            "data_rollback": [
                "Restore database from backup",
                "Verify data integrity",
                "Update system configurations",
                "Test system functionality"
            ],
            "full_rollback": [
                "Revert to previous system version",
                "Restore all configurations",
                "Validate all integrations",
                "Conduct full system testing",
                "Notify all stakeholders"
            ]
        }
        
        return procedures
        
    def save_configuration(self):
        """Save site preparation configuration to file."""
        config = {
            "technical_requirements": asdict(self.requirements),
            "hospital_sites": {
                site_id: asdict(site) for site_id, site in self.sites.items()
            }
        }
        
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Site preparation configuration saved to {self.config_path}")
        
    def load_configuration(self):
        """Load site preparation configuration from file."""
        if not Path(self.config_path).exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if "technical_requirements" in config:
            self.requirements = TechnicalRequirements(**config["technical_requirements"])
            
        if "hospital_sites" in config:
            for site_id, site_data in config["hospital_sites"].items():
                self.sites[site_id] = HospitalSite(**site_data)
                
        logger.info(f"Site preparation configuration loaded from {self.config_path}")

# Example usage and demo sites
def create_demo_sites() -> SitePreparationManager:
    """Create demo hospital sites for testing."""
    manager = SitePreparationManager()
    
    # Demo site 1: Large academic medical center
    site1 = HospitalSite(
        site_id="AMC001",
        name="University Medical Center",
        location="Boston, MA",
        contact_email="it@umc.edu",
        it_contact="John Smith",
        pathology_volume=8000,
        current_systems=["Epic EMR", "Sunquest LIS", "Leica Scanner"],
        integration_requirements=["Epic FHIR", "Sunquest HL7", "DICOM"],
        compliance_requirements=["HIPAA", "SOX", "Joint Commission"]
    )
    
    # Demo site 2: Community hospital
    site2 = HospitalSite(
        site_id="CH001", 
        name="Community General Hospital",
        location="Denver, CO",
        contact_email="support@cgh.org",
        it_contact="Sarah Johnson",
        pathology_volume=2500,
        current_systems=["Cerner EMR", "Cerner PathNet LIS"],
        integration_requirements=["Cerner FHIR", "PathNet HL7"],
        compliance_requirements=["HIPAA", "State Regulations"]
    )
    
    # Demo site 3: Regional medical center
    site3 = HospitalSite(
        site_id="RMC001",
        name="Regional Medical Center",
        location="Atlanta, GA", 
        contact_email="tech@rmc.com",
        it_contact="Mike Davis",
        pathology_volume=4500,
        current_systems=["Allscripts EMR", "Custom LIS", "Hamamatsu Scanner"],
        integration_requirements=["Allscripts API", "Custom HL7", "DICOM"],
        compliance_requirements=["HIPAA", "Medicare", "Medicaid"]
    )
    
    manager.add_hospital_site(site1)
    manager.add_hospital_site(site2)
    manager.add_hospital_site(site3)
    
    return manager

if __name__ == "__main__":
    # Demo usage
    manager = create_demo_sites()
    
    # Assess requirements for each site
    for site_id in manager.sites:
        assessment = manager.assess_technical_requirements(site_id)
        print(f"\nTechnical Assessment for {assessment['site_name']}:")
        print(f"Recommended CPU cores: {assessment['recommended_specs']['cpu_cores']}")
        print(f"Recommended RAM: {assessment['recommended_specs']['ram_gb']} GB")
        print(f"Estimated setup time: {assessment['estimated_setup_time_days']} days")
        
        # Generate integration plan
        plan = manager.create_integration_plan(site_id)
        print(f"Integration phases: {len(plan['integration_phases'])}")
        
        # Generate training program
        training = manager.generate_training_program(site_id)
        print(f"Training modules: {len(training['training_modules'])}")
        
    manager.save_configuration()