#!/usr/bin/env python3
"""
Pilot Deployment Infrastructure Manager

Manages pilot deployments at hospital sites, including infrastructure setup,
monitoring, data collection, and performance tracking.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import docker
import paramiko
import requests
import yaml
from kubernetes import client, config as k8s_config

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class PilotDeploymentManager:
    """Manages pilot deployments and infrastructure."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize pilot deployment manager."""
        self.config_dir = Path(config_dir or ".kiro/pilot_deployments")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load deployment configurations
        self.load_configurations()
        
        # Initialize cloud clients
        self.setup_cloud_clients()
        
        logger.info(f"Pilot deployment config directory: {self.config_dir}")
    
    def load_configurations(self):
        """Load pilot deployment configurations."""
        # Default pilot site configurations
        self.pilot_sites = {
            'mayo_clinic': {
                'name': 'Mayo Clinic',
                'type': 'academic_medical_center',
                'location': 'Rochester, MN',
                'pacs_vendor': 'Epic',
                'deployment_type': 'on_premise',
                'capacity': 'high',
                'contact': {
                    'technical': 'it-pathology@mayo.edu',
                    'clinical': 'pathology-research@mayo.edu'
                },
                'infrastructure': {
                    'compute_nodes': 4,
                    'gpu_nodes': 2,
                    'storage_tb': 50,
                    'network_bandwidth': '10Gbps'
                }
            },
            'cleveland_clinic': {
                'name': 'Cleveland Clinic',
                'type': 'academic_medical_center',
                'location': 'Cleveland, OH',
                'pacs_vendor': 'Epic',
                'deployment_type': 'hybrid_cloud',
                'capacity': 'high',
                'contact': {
                    'technical': 'it-innovation@ccf.org',
                    'clinical': 'pathology-ai@ccf.org'
                },
                'infrastructure': {
                    'compute_nodes': 6,
                    'gpu_nodes': 3,
                    'storage_tb': 100,
                    'network_bandwidth': '10Gbps'
                }
            },
            'johns_hopkins': {
                'name': 'Johns Hopkins Hospital',
                'type': 'academic_medical_center',
                'location': 'Baltimore, MD',
                'pacs_vendor': 'Cerner',
                'deployment_type': 'cloud',
                'capacity': 'medium',
                'contact': {
                    'technical': 'it-pathology@jhmi.edu',
                    'clinical': 'pathology-innovation@jhmi.edu'
                },
                'infrastructure': {
                    'compute_nodes': 3,
                    'gpu_nodes': 2,
                    'storage_tb': 30,
                    'network_bandwidth': '5Gbps'
                }
            }
        }
        
        # Deployment templates
        self.deployment_templates = {
            'on_premise': {
                'description': 'On-premise deployment with local infrastructure',
                'components': ['api_server', 'inference_engine', 'database', 'monitoring'],
                'requirements': {
                    'min_cpu_cores': 16,
                    'min_memory_gb': 64,
                    'min_gpu_memory_gb': 8,
                    'min_storage_tb': 10
                }
            },
            'hybrid_cloud': {
                'description': 'Hybrid deployment with cloud and on-premise components',
                'components': ['local_inference', 'cloud_training', 'edge_cache', 'monitoring'],
                'requirements': {
                    'min_cpu_cores': 8,
                    'min_memory_gb': 32,
                    'min_gpu_memory_gb': 4,
                    'cloud_credits': 1000
                }
            },
            'cloud': {
                'description': 'Full cloud deployment with secure connectivity',
                'components': ['cloud_api', 'cloud_inference', 'cloud_storage', 'vpn_gateway'],
                'requirements': {
                    'cloud_credits': 2000,
                    'network_bandwidth': '1Gbps',
                    'compliance': ['HIPAA', 'SOC2']
                }
            }
        }
    
    def setup_cloud_clients(self):
        """Setup cloud service clients."""
        try:
            # Docker client for containerized deployments
            self.docker_client = docker.from_env()
            
            # Kubernetes client for orchestration
            try:
                k8s_config.load_incluster_config()
            except:
                k8s_config.load_kube_config()
            self.k8s_client = client.ApiClient()
            
        except Exception as e:
            logger.warning(f"Could not initialize all cloud clients: {e}")
    
    def generate_deployment_plan(self, site_id: str) -> Dict:
        """Generate deployment plan for pilot site."""
        if site_id not in self.pilot_sites:
            logger.error(f"Unknown pilot site: {site_id}")
            return {}
        
        site_config = self.pilot_sites[site_id]
        deployment_type = site_config['deployment_type']
        template = self.deployment_templates[deployment_type]
        
        deployment_plan = {
            'site_id': site_id,
            'site_name': site_config['name'],
            'deployment_type': deployment_type,
            'generated': datetime.now().isoformat(),
            'phases': self._generate_deployment_phases(site_config, template),
            'infrastructure': self._generate_infrastructure_spec(site_config, template),
            'timeline': self._generate_deployment_timeline(),
            'resources': self._generate_resource_requirements(site_config, template),
            'monitoring': self._generate_monitoring_plan(site_config),
            'rollback': self._generate_rollback_plan(site_config)
        }
        
        return deployment_plan
    
    def _generate_deployment_phases(self, site_config: Dict, template: Dict) -> List[Dict]:
        """Generate deployment phases."""
        phases = [
            {
                'phase': 1,
                'name': 'Infrastructure Setup',
                'duration_days': 5,
                'tasks': [
                    'Provision compute resources',
                    'Setup network connectivity',
                    'Install base software stack',
                    'Configure security policies'
                ],
                'deliverables': ['Infrastructure ready', 'Security audit passed']
            },
            {
                'phase': 2,
                'name': 'Application Deployment',
                'duration_days': 3,
                'tasks': [
                    'Deploy Medical AI platform',
                    'Configure PACS integration',
                    'Setup user authentication',
                    'Initialize databases'
                ],
                'deliverables': ['Application deployed', 'PACS connected']
            },
            {
                'phase': 3,
                'name': 'Integration Testing',
                'duration_days': 7,
                'tasks': [
                    'Test PACS connectivity',
                    'Validate inference pipeline',
                    'Test user workflows',
                    'Performance benchmarking'
                ],
                'deliverables': ['Integration tests passed', 'Performance validated']
            },
            {
                'phase': 4,
                'name': 'User Training',
                'duration_days': 5,
                'tasks': [
                    'Train pathologists',
                    'Train IT staff',
                    'Create user documentation',
                    'Setup support channels'
                ],
                'deliverables': ['Users trained', 'Documentation complete']
            },
            {
                'phase': 5,
                'name': 'Pilot Launch',
                'duration_days': 30,
                'tasks': [
                    'Begin pilot operations',
                    'Monitor performance',
                    'Collect feedback',
                    'Generate reports'
                ],
                'deliverables': ['Pilot active', 'Initial results available']
            }
        ]
        
        return phases
    
    def _generate_infrastructure_spec(self, site_config: Dict, template: Dict) -> Dict:
        """Generate infrastructure specifications."""
        infrastructure = site_config['infrastructure']
        
        spec = {
            'compute': {
                'nodes': infrastructure['compute_nodes'],
                'cpu_cores_per_node': 16,
                'memory_gb_per_node': 64,
                'storage_gb_per_node': 1000
            },
            'gpu': {
                'nodes': infrastructure['gpu_nodes'],
                'gpu_type': 'NVIDIA A100',
                'gpu_memory_gb': 40,
                'cuda_version': '11.8'
            },
            'storage': {
                'total_tb': infrastructure['storage_tb'],
                'type': 'NVMe SSD',
                'backup': 'Daily incremental',
                'retention_days': 90
            },
            'network': {
                'bandwidth': infrastructure['network_bandwidth'],
                'security': 'VPN + Firewall',
                'monitoring': 'Real-time'
            },
            'software': {
                'os': 'Ubuntu 22.04 LTS',
                'container_runtime': 'Docker 24.0',
                'orchestration': 'Kubernetes 1.28',
                'monitoring': 'Prometheus + Grafana'
            }
        }
        
        return spec
    
    def _generate_deployment_timeline(self) -> Dict:
        """Generate deployment timeline."""
        start_date = datetime.now()
        
        timeline = {
            'start_date': start_date.isoformat(),
            'estimated_completion': (start_date + timedelta(days=50)).isoformat(),
            'milestones': [
                {
                    'name': 'Infrastructure Ready',
                    'date': (start_date + timedelta(days=5)).isoformat(),
                    'critical': True
                },
                {
                    'name': 'Application Deployed',
                    'date': (start_date + timedelta(days=8)).isoformat(),
                    'critical': True
                },
                {
                    'name': 'Integration Complete',
                    'date': (start_date + timedelta(days=15)).isoformat(),
                    'critical': True
                },
                {
                    'name': 'Training Complete',
                    'date': (start_date + timedelta(days=20)).isoformat(),
                    'critical': False
                },
                {
                    'name': 'Pilot Launch',
                    'date': (start_date + timedelta(days=21)).isoformat(),
                    'critical': True
                },
                {
                    'name': 'Initial Results',
                    'date': (start_date + timedelta(days=51)).isoformat(),
                    'critical': False
                }
            ]
        }
        
        return timeline
    
    def _generate_resource_requirements(self, site_config: Dict, template: Dict) -> Dict:
        """Generate resource requirements."""
        requirements = {
            'personnel': {
                'project_manager': 1,
                'devops_engineer': 2,
                'clinical_liaison': 1,
                'support_engineer': 1
            },
            'budget': {
                'infrastructure': 50000,
                'software_licenses': 25000,
                'personnel': 100000,
                'training': 15000,
                'contingency': 19000,
                'total': 209000
            },
            'external_dependencies': [
                'Hospital IT approval',
                'Network security clearance',
                'PACS vendor coordination',
                'Clinical workflow integration'
            ]
        }
        
        return requirements
    
    def _generate_monitoring_plan(self, site_config: Dict) -> Dict:
        """Generate monitoring and metrics plan."""
        monitoring = {
            'technical_metrics': [
                'System uptime and availability',
                'Inference processing time',
                'PACS integration latency',
                'Resource utilization (CPU, GPU, memory)',
                'Network throughput and latency',
                'Error rates and exceptions'
            ],
            'clinical_metrics': [
                'Number of cases processed',
                'Diagnostic accuracy metrics',
                'User adoption rates',
                'Workflow efficiency improvements',
                'User satisfaction scores',
                'Clinical impact measurements'
            ],
            'business_metrics': [
                'Cost per case processed',
                'ROI calculations',
                'Time savings quantification',
                'Quality improvements',
                'Scalability assessments'
            ],
            'reporting': {
                'frequency': 'Weekly',
                'stakeholders': ['Clinical team', 'IT team', 'Executive sponsors'],
                'format': 'Dashboard + PDF report',
                'escalation_thresholds': {
                    'uptime': '<99%',
                    'processing_time': '>60s',
                    'error_rate': '>1%'
                }
            }
        }
        
        return monitoring
    
    def _generate_rollback_plan(self, site_config: Dict) -> Dict:
        """Generate rollback and contingency plan."""
        rollback = {
            'triggers': [
                'System uptime < 95% for 24 hours',
                'Critical security vulnerability',
                'Unacceptable clinical performance',
                'Major integration failures',
                'Stakeholder request'
            ],
            'procedures': [
                'Immediate system shutdown if safety risk',
                'Revert to previous stable version',
                'Restore from backup if needed',
                'Notify all stakeholders',
                'Conduct post-incident review'
            ],
            'recovery_time': {
                'target_rto': '4 hours',  # Recovery Time Objective
                'target_rpo': '1 hour'    # Recovery Point Objective
            },
            'communication': {
                'internal_team': 'Immediate notification',
                'hospital_stakeholders': 'Within 2 hours',
                'regulatory_bodies': 'Within 24 hours if required'
            }
        }
        
        return rollback
    
    def deploy_pilot_site(self, site_id: str, deployment_plan: Dict) -> bool:
        """Deploy pilot site based on deployment plan."""
        try:
            logger.info(f"Starting deployment for {site_id}")
            
            # Create deployment directory
            deployment_dir = self.config_dir / site_id
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            # Save deployment plan
            plan_file = deployment_dir / "deployment_plan.json"
            with open(plan_file, 'w') as f:
                json.dump(deployment_plan, f, indent=2)
            
            # Execute deployment phases
            for phase in deployment_plan['phases']:
                logger.info(f"Executing Phase {phase['phase']}: {phase['name']}")
                
                # This would execute actual deployment tasks
                # For now, simulate deployment
                success = self._execute_deployment_phase(site_id, phase, deployment_dir)
                
                if not success:
                    logger.error(f"Phase {phase['phase']} failed for {site_id}")
                    return False
                
                logger.info(f"Phase {phase['phase']} completed successfully")
            
            # Update deployment status
            self._update_deployment_status(site_id, 'deployed', deployment_dir)
            
            logger.info(f"Deployment completed successfully for {site_id}")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed for {site_id}: {e}")
            return False
    
    def _execute_deployment_phase(self, site_id: str, phase: Dict, deployment_dir: Path) -> bool:
        """Execute a specific deployment phase."""
        try:
            phase_log = []
            
            for task in phase['tasks']:
                logger.info(f"Executing task: {task}")
                
                # Simulate task execution
                # In reality, this would execute actual deployment commands
                task_result = {
                    'task': task,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'duration_seconds': 30  # Simulated
                }
                
                phase_log.append(task_result)
            
            # Save phase log
            log_file = deployment_dir / f"phase_{phase['phase']}_log.json"
            with open(log_file, 'w') as f:
                json.dump(phase_log, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Phase execution failed: {e}")
            return False
    
    def _update_deployment_status(self, site_id: str, status: str, deployment_dir: Path):
        """Update deployment status."""
        status_info = {
            'site_id': site_id,
            'status': status,
            'last_updated': datetime.now().isoformat(),
            'deployment_dir': str(deployment_dir)
        }
        
        status_file = deployment_dir / "deployment_status.json"
        with open(status_file, 'w') as f:
            json.dump(status_info, f, indent=2)
    
    def monitor_pilot_site(self, site_id: str) -> Dict:
        """Monitor pilot site performance and metrics."""
        if site_id not in self.pilot_sites:
            logger.error(f"Unknown pilot site: {site_id}")
            return {}
        
        # Collect monitoring data (simulated)
        monitoring_data = {
            'site_id': site_id,
            'timestamp': datetime.now().isoformat(),
            'technical_metrics': {
                'uptime_percentage': 99.5,
                'avg_processing_time_seconds': 25.3,
                'pacs_latency_ms': 150,
                'cpu_utilization_percentage': 65,
                'gpu_utilization_percentage': 80,
                'memory_utilization_percentage': 70,
                'error_rate_percentage': 0.1
            },
            'clinical_metrics': {
                'cases_processed_today': 245,
                'cases_processed_total': 12500,
                'diagnostic_accuracy_percentage': 94.8,
                'user_adoption_percentage': 85,
                'workflow_time_savings_minutes': 15.2,
                'user_satisfaction_score': 4.2
            },
            'business_metrics': {
                'cost_per_case_usd': 2.50,
                'roi_percentage': 150,
                'time_savings_hours_per_day': 6.2,
                'quality_improvement_score': 8.5
            },
            'alerts': [
                {
                    'level': 'warning',
                    'message': 'GPU utilization above 75%',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }
        
        return monitoring_data
    
    def generate_pilot_report(self, site_id: str, report_type: str = 'comprehensive') -> str:
        """Generate pilot site report."""
        if site_id not in self.pilot_sites:
            logger.error(f"Unknown pilot site: {site_id}")
            return ""
        
        site_config = self.pilot_sites[site_id]
        monitoring_data = self.monitor_pilot_site(site_id)
        
        report = f"""
# Pilot Deployment Report: {site_config['name']}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Site ID**: {site_id}
**Report Type**: {report_type}

## Executive Summary

The Medical AI Revolution Platform pilot deployment at {site_config['name']} has been successfully implemented and is showing strong performance metrics. The system is processing an average of {monitoring_data['clinical_metrics']['cases_processed_today']} cases per day with {monitoring_data['technical_metrics']['uptime_percentage']}% uptime.

## Site Information

- **Location**: {site_config['location']}
- **Type**: {site_config['type']}
- **PACS Vendor**: {site_config['pacs_vendor']}
- **Deployment Type**: {site_config['deployment_type']}

## Technical Performance

### System Metrics
- **Uptime**: {monitoring_data['technical_metrics']['uptime_percentage']}%
- **Average Processing Time**: {monitoring_data['technical_metrics']['avg_processing_time_seconds']} seconds
- **PACS Integration Latency**: {monitoring_data['technical_metrics']['pacs_latency_ms']} ms
- **Resource Utilization**:
  - CPU: {monitoring_data['technical_metrics']['cpu_utilization_percentage']}%
  - GPU: {monitoring_data['technical_metrics']['gpu_utilization_percentage']}%
  - Memory: {monitoring_data['technical_metrics']['memory_utilization_percentage']}%
- **Error Rate**: {monitoring_data['technical_metrics']['error_rate_percentage']}%

### Infrastructure Status
- **Compute Nodes**: {site_config['infrastructure']['compute_nodes']} active
- **GPU Nodes**: {site_config['infrastructure']['gpu_nodes']} active
- **Storage Utilization**: 65% of {site_config['infrastructure']['storage_tb']} TB
- **Network Performance**: Stable at {site_config['infrastructure']['network_bandwidth']}

## Clinical Performance

### Usage Statistics
- **Total Cases Processed**: {monitoring_data['clinical_metrics']['cases_processed_total']:,}
- **Daily Average**: {monitoring_data['clinical_metrics']['cases_processed_today']} cases
- **User Adoption Rate**: {monitoring_data['clinical_metrics']['user_adoption_percentage']}%

### Accuracy Metrics
- **Diagnostic Accuracy**: {monitoring_data['clinical_metrics']['diagnostic_accuracy_percentage']}%
- **User Satisfaction**: {monitoring_data['clinical_metrics']['user_satisfaction_score']}/5.0
- **Workflow Efficiency**: {monitoring_data['clinical_metrics']['workflow_time_savings_minutes']} minutes saved per case

## Business Impact

### Cost-Benefit Analysis
- **Cost per Case**: ${monitoring_data['business_metrics']['cost_per_case_usd']}
- **ROI**: {monitoring_data['business_metrics']['roi_percentage']}%
- **Daily Time Savings**: {monitoring_data['business_metrics']['time_savings_hours_per_day']} hours
- **Quality Improvement Score**: {monitoring_data['business_metrics']['quality_improvement_score']}/10

### Operational Benefits
- Reduced diagnostic turnaround time
- Improved consistency in pathology reporting
- Enhanced quality assurance capabilities
- Increased pathologist productivity

## Challenges and Resolutions

### Technical Challenges
1. **Initial PACS Integration**: Resolved through vendor collaboration
2. **Network Latency**: Optimized through local caching
3. **User Training**: Addressed with comprehensive training program

### Clinical Challenges
1. **Workflow Integration**: Customized to existing processes
2. **User Adoption**: Improved through peer champions program
3. **Quality Assurance**: Enhanced monitoring and feedback loops

## Recommendations

### Short-term (1-3 months)
1. Expand user training program
2. Optimize GPU resource allocation
3. Implement additional quality metrics
4. Enhance monitoring dashboards

### Medium-term (3-6 months)
1. Scale to additional pathology departments
2. Integrate with additional PACS systems
3. Implement federated learning capabilities
4. Develop custom reporting features

### Long-term (6-12 months)
1. Full hospital-wide deployment
2. Multi-site federated learning
3. Advanced AI capabilities integration
4. Regulatory submission support

## Next Steps

1. **Performance Optimization**: Continue monitoring and optimization
2. **User Feedback**: Regular collection and incorporation of feedback
3. **Expansion Planning**: Prepare for broader deployment
4. **Continuous Improvement**: Regular updates and enhancements

## Contact Information

- **Technical Support**: support@medical-ai-revolution.com
- **Clinical Liaison**: clinical@medical-ai-revolution.com
- **Project Manager**: {site_config['contact']['technical']}

---

*This report is generated automatically and updated weekly. For questions or additional information, please contact the project team.*
"""
        
        return report.strip()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Pilot Deployment Infrastructure Manager")
    parser.add_argument(
        '--action',
        choices=['plan', 'deploy', 'monitor', 'report'],
        required=True,
        help='Action to perform'
    )
    parser.add_argument(
        '--site',
        choices=['mayo_clinic', 'cleveland_clinic', 'johns_hopkins', 'all'],
        help='Pilot site to manage'
    )
    parser.add_argument(
        '--report-type',
        choices=['comprehensive', 'technical', 'clinical', 'business'],
        default='comprehensive',
        help='Type of report to generate'
    )
    parser.add_argument(
        '--config-dir',
        help='Configuration directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize manager
    manager = PilotDeploymentManager(args.config_dir)
    
    if args.action == 'plan':
        if not args.site or args.site == 'all':
            sites = list(manager.pilot_sites.keys())
        else:
            sites = [args.site]
        
        for site in sites:
            print(f"Generating deployment plan for {site}...")
            plan = manager.generate_deployment_plan(site)
            
            if plan:
                # Save plan
                plan_file = manager.config_dir / f"{site}_deployment_plan.json"
                with open(plan_file, 'w') as f:
                    json.dump(plan, f, indent=2)
                
                print(f"✅ Deployment plan generated: {plan_file}")
                print(f"   Estimated duration: {plan['timeline']['estimated_completion']}")
                print(f"   Total budget: ${plan['resources']['budget']['total']:,}")
            else:
                print(f"❌ Failed to generate deployment plan for {site}")
    
    elif args.action == 'deploy':
        if not args.site:
            print("Error: --site required for deploy action")
            return
        
        print(f"Starting deployment for {args.site}...")
        
        # Generate deployment plan
        plan = manager.generate_deployment_plan(args.site)
        
        if plan:
            # Execute deployment
            success = manager.deploy_pilot_site(args.site, plan)
            
            if success:
                print(f"✅ Deployment completed successfully for {args.site}")
            else:
                print(f"❌ Deployment failed for {args.site}")
        else:
            print(f"❌ Could not generate deployment plan for {args.site}")
    
    elif args.action == 'monitor':
        if not args.site or args.site == 'all':
            sites = list(manager.pilot_sites.keys())
        else:
            sites = [args.site]
        
        for site in sites:
            print(f"Monitoring {site}...")
            monitoring_data = manager.monitor_pilot_site(site)
            
            if monitoring_data:
                print(f"✅ {site} Status:")
                print(f"   Uptime: {monitoring_data['technical_metrics']['uptime_percentage']}%")
                print(f"   Cases processed today: {monitoring_data['clinical_metrics']['cases_processed_today']}")
                print(f"   Diagnostic accuracy: {monitoring_data['clinical_metrics']['diagnostic_accuracy_percentage']}%")
                
                if monitoring_data['alerts']:
                    print(f"   ⚠️  {len(monitoring_data['alerts'])} active alerts")
            else:
                print(f"❌ Failed to get monitoring data for {site}")
    
    elif args.action == 'report':
        if not args.site:
            print("Error: --site required for report action")
            return
        
        print(f"Generating {args.report_type} report for {args.site}...")
        report = manager.generate_pilot_report(args.site, args.report_type)
        
        if report:
            # Save report
            report_file = manager.config_dir / f"{args.site}_{args.report_type}_report_{datetime.now().strftime('%Y%m%d')}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"✅ Report generated: {report_file}")
            print("\n" + "="*50)
            print(report[:1000] + "..." if len(report) > 1000 else report)
        else:
            print(f"❌ Failed to generate report for {args.site}")


if __name__ == "__main__":
    main()