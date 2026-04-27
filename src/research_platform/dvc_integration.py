"""
DVC Integration

Data Version Control for datasets.
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging


class DVCManager:
    """
    DVC integration for dataset versioning
    
    Features:
    - Dataset snapshots
    - Version tracking
    - Reproducible experiments
    - Lineage tracking
    """
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.logger = logging.getLogger(__name__)
    
    def init(self) -> bool:
        """Initialize DVC repo"""
        
        try:
            subprocess.run(['dvc', 'init'], cwd=self.repo_path, check=True)
            self.logger.info("DVC initialized")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"DVC init failed: {e}")
            return False
    
    def add_dataset(self, dataset_path: Path) -> bool:
        """Add dataset to DVC"""
        
        try:
            subprocess.run(['dvc', 'add', str(dataset_path)], 
                         cwd=self.repo_path, check=True)
            self.logger.info(f"Added to DVC: {dataset_path}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"DVC add failed: {e}")
            return False
    
    def commit_snapshot(self, message: str, tag: Optional[str] = None) -> bool:
        """Commit dataset snapshot"""
        
        try:
            # Git commit
            subprocess.run(['git', 'add', '.'], cwd=self.repo_path, check=True)
            subprocess.run(['git', 'commit', '-m', message], 
                         cwd=self.repo_path, check=True)
            
            # Tag
            if tag:
                subprocess.run(['git', 'tag', tag], cwd=self.repo_path, check=True)
            
            self.logger.info(f"Snapshot committed: {message}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Commit failed: {e}")
            return False
    
    def checkout_version(self, version: str) -> bool:
        """Checkout dataset version"""
        
        try:
            subprocess.run(['git', 'checkout', version], 
                         cwd=self.repo_path, check=True)
            subprocess.run(['dvc', 'checkout'], 
                         cwd=self.repo_path, check=True)
            self.logger.info(f"Checked out version: {version}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Checkout failed: {e}")
            return False
    
    def list_versions(self) -> List[str]:
        """List dataset versions"""
        
        try:
            result = subprocess.run(['git', 'tag'], 
                                  cwd=self.repo_path, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            return result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return []
    
    def push_remote(self, remote: str = 'origin') -> bool:
        """Push to remote"""
        
        try:
            subprocess.run(['git', 'push', remote], 
                         cwd=self.repo_path, check=True)
            subprocess.run(['dvc', 'push'], 
                         cwd=self.repo_path, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Push failed: {e}")
            return False
    
    def pull_remote(self, remote: str = 'origin') -> bool:
        """Pull from remote"""
        
        try:
            subprocess.run(['git', 'pull', remote], 
                         cwd=self.repo_path, check=True)
            subprocess.run(['dvc', 'pull'], 
                         cwd=self.repo_path, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Pull failed: {e}")
            return False
