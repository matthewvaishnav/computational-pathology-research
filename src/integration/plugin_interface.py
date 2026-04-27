"""
Plugin Interface for Medical AI Integration Ecosystem

Define plugin interfaces for scanner, LIS, EMR, cloud integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

class PluginType(Enum):
    """Plugin types"""
    SCANNER = "scanner"
    LIS = "lis"
    EMR = "emr"
    CLOUD = "cloud"
    STORAGE = "storage"
    ANALYTICS = "analytics"

class PluginStatus(Enum):
    """Plugin status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    INITIALIZING = "initializing"

@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    plugin_type: PluginType
    vendor: str
    description: str
    capabilities: List[str]
    config_schema: Dict[str, Any]
    dependencies: List[str] = None

@dataclass
class ImageMetadata:
    """WSI image metadata"""
    slide_id: str
    patient_id: str
    case_id: str
    acquisition_date: datetime
    scanner_model: str
    magnification: float
    width: int
    height: int
    tile_size: int
    format: str
    compression: str
    metadata: Dict[str, Any] = None

class BasePlugin(ABC):
    """Base plugin interface"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.status = PluginStatus.INITIALIZING
        self.metadata = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """Init plugin. Return success."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown plugin. Return success."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Health check. Return status dict."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    def get_status(self) -> PluginStatus:
        """Get plugin status."""
        return self.status

class ScannerPlugin(BasePlugin):
    """Scanner integration plugin interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to scanner. Return success."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from scanner. Return success."""
        pass
    
    @abstractmethod
    def get_slide_list(self) -> List[str]:
        """Get available slide IDs."""
        pass
    
    @abstractmethod
    def get_slide_metadata(self, slide_id: str) -> ImageMetadata:
        """Get slide metadata."""
        pass
    
    @abstractmethod
    def read_region(
        self, 
        slide_id: str, 
        x: int, 
        y: int, 
        width: int, 
        height: int,
        level: int = 0
    ) -> np.ndarray:
        """Read image region. Return RGB array."""
        pass
    
    @abstractmethod
    def get_thumbnail(self, slide_id: str, max_size: int = 512) -> np.ndarray:
        """Get slide thumbnail."""
        pass

class LISPlugin(BasePlugin):
    """LIS integration plugin interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to LIS."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from LIS."""
        pass
    
    @abstractmethod
    def get_case(self, case_id: str) -> Dict[str, Any]:
        """Get case data."""
        pass
    
    @abstractmethod
    def create_case(self, case_data: Dict[str, Any]) -> str:
        """Create case. Return case ID."""
        pass
    
    @abstractmethod
    def update_case(self, case_id: str, updates: Dict[str, Any]) -> bool:
        """Update case. Return success."""
        pass
    
    @abstractmethod
    def submit_result(self, case_id: str, result: Dict[str, Any]) -> bool:
        """Submit AI result. Return success."""
        pass
    
    @abstractmethod
    def get_worklist(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get worklist."""
        pass

class EMRPlugin(BasePlugin):
    """EMR integration plugin interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to EMR."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from EMR."""
        pass
    
    @abstractmethod
    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """Get patient data."""
        pass
    
    @abstractmethod
    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient history."""
        pass
    
    @abstractmethod
    def create_note(self, patient_id: str, note: Dict[str, Any]) -> str:
        """Create clinical note. Return note ID."""
        pass
    
    @abstractmethod
    def get_orders(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient orders."""
        pass
    
    @abstractmethod
    def send_result(self, patient_id: str, result: Dict[str, Any]) -> bool:
        """Send result to EMR. Return success."""
        pass

class CloudPlugin(BasePlugin):
    """Cloud platform integration plugin interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to cloud."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from cloud."""
        pass
    
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file. Return success."""
        pass
    
    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file. Return success."""
        pass
    
    @abstractmethod
    def list_files(self, path: str) -> List[str]:
        """List files at path."""
        pass
    
    @abstractmethod
    def delete_file(self, path: str) -> bool:
        """Delete file. Return success."""
        pass
    
    @abstractmethod
    def invoke_function(self, function_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke cloud function. Return result."""
        pass

class StoragePlugin(BasePlugin):
    """Storage plugin interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to storage."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from storage."""
        pass
    
    @abstractmethod
    def store(self, key: str, data: bytes) -> bool:
        """Store data. Return success."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> bytes:
        """Retrieve data."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data. Return success."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with prefix."""
        pass

class AnalyticsPlugin(BasePlugin):
    """Analytics plugin interface"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to analytics."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from analytics."""
        pass
    
    @abstractmethod
    def log_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Log event. Return success."""
        pass
    
    @abstractmethod
    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> bool:
        """Log metric. Return success."""
        pass
    
    @abstractmethod
    def query_metrics(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Query metrics."""
        pass
    
    @abstractmethod
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create dashboard. Return dashboard ID."""
        pass

# Plugin registry
class PluginRegistry:
    """Plugin registry"""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
    
    def register(self, name: str, plugin: BasePlugin) -> bool:
        """Register plugin."""
        if name in self.plugins:
            return False
        self.plugins[name] = plugin
        return True
    
    def unregister(self, name: str) -> bool:
        """Unregister plugin."""
        if name not in self.plugins:
            return False
        plugin = self.plugins[name]
        plugin.shutdown()
        del self.plugins[name]
        return True
    
    def get(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List registered plugins."""
        return list(self.plugins.keys())
    
    def get_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get plugins by type."""
        return [
            p for p in self.plugins.values() 
            if p.get_metadata().plugin_type == plugin_type
        ]