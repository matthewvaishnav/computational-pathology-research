"""
Plugin Security Sandboxing

Isolate plugins, limit resources, validate operations.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
import resource
import signal
from functools import wraps

logger = logging.getLogger(__name__)

class Permission(Enum):
    """Plugin permissions"""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    NETWORK = "network"
    DATABASE = "database"
    SYSTEM = "system"
    PATIENT_DATA = "patient_data"

@dataclass
class ResourceLimits:
    """Resource limits for plugin"""
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_execution_time_sec: int = 300
    max_file_size_mb: int = 100
    max_network_requests: int = 1000

@dataclass
class SecurityPolicy:
    """Security policy for plugin"""
    permissions: List[Permission]
    resource_limits: ResourceLimits
    allowed_hosts: List[str] = None
    allowed_paths: List[str] = None
    require_encryption: bool = True
    audit_all_operations: bool = True

class PluginSandbox:
    """Plugin sandbox"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.operation_count: Dict[str, int] = {}
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def check_permission(self, permission: Permission) -> bool:
        """Check if plugin has permission."""
        return permission in self.policy.permissions
    
    def enforce_resource_limits(self):
        """Enforce resource limits."""
        try:
            # Memory limit
            if self.policy.resource_limits.max_memory_mb:
                max_mem_bytes = self.policy.resource_limits.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
            
            # CPU time limit
            if self.policy.resource_limits.max_execution_time_sec:
                max_cpu = self.policy.resource_limits.max_execution_time_sec
                resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu))
            
        except Exception as e:
            logger.error(f"Set resource limits fail: {e}")
    
    def validate_file_access(self, path: str, write: bool = False) -> bool:
        """Validate file access."""
        # Check permission
        required_perm = Permission.FILE_WRITE if write else Permission.FILE_READ
        if not self.check_permission(required_perm):
            logger.warning(f"Plugin lacks {required_perm.value} permission")
            return False
        
        # Check allowed paths
        if self.policy.allowed_paths:
            if not any(path.startswith(allowed) for allowed in self.policy.allowed_paths):
                logger.warning(f"Path {path} not in allowed paths")
                return False
        
        return True
    
    def validate_network_access(self, host: str) -> bool:
        """Validate network access."""
        # Check permission
        if not self.check_permission(Permission.NETWORK):
            logger.warning("Plugin lacks network permission")
            return False
        
        # Check allowed hosts
        if self.policy.allowed_hosts:
            if host not in self.policy.allowed_hosts:
                logger.warning(f"Host {host} not in allowed hosts")
                return False
        
        # Check request limit
        with self.lock:
            count = self.operation_count.get('network_requests', 0)
            if count >= self.policy.resource_limits.max_network_requests:
                logger.warning("Network request limit exceeded")
                return False
            self.operation_count['network_requests'] = count + 1
        
        return True
    
    def validate_patient_data_access(self, patient_id: str) -> bool:
        """Validate patient data access."""
        if not self.check_permission(Permission.PATIENT_DATA):
            logger.warning("Plugin lacks patient data permission")
            return False
        
        # Audit
        if self.policy.audit_all_operations:
            self._audit_operation('patient_data_access', {'patient_id': patient_id})
        
        return True
    
    def _audit_operation(self, operation: str, details: Dict[str, Any]):
        """Audit operation."""
        logger.info(f"AUDIT: {operation} - {details}")
    
    def wrap_function(self, func: Callable, operation_name: str) -> Callable:
        """Wrap function with sandbox checks."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check execution time
            elapsed = time.time() - self.start_time
            if elapsed > self.policy.resource_limits.max_execution_time_sec:
                raise TimeoutError(f"Plugin execution time limit exceeded")
            
            # Audit
            if self.policy.audit_all_operations:
                self._audit_operation(operation_name, {
                    'args': str(args)[:100],
                    'kwargs': str(kwargs)[:100]
                })
            
            # Execute
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Sandboxed function {operation_name} fail: {e}")
                raise
        
        return wrapper
    
    def create_restricted_environment(self) -> Dict[str, Any]:
        """Create restricted execution environment."""
        # Minimal safe builtins
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'range': range,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        }
        
        return {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }

class SandboxedPluginWrapper:
    """Wrapper for sandboxed plugin"""
    
    def __init__(self, plugin: Any, sandbox: PluginSandbox):
        self.plugin = plugin
        self.sandbox = sandbox
    
    def __getattr__(self, name: str):
        """Intercept attribute access."""
        attr = getattr(self.plugin, name)
        
        # Wrap methods
        if callable(attr):
            return self.sandbox.wrap_function(attr, f"{self.plugin.__class__.__name__}.{name}")
        
        return attr

def create_sandbox(policy: SecurityPolicy) -> PluginSandbox:
    """Create plugin sandbox."""
    sandbox = PluginSandbox(policy)
    sandbox.enforce_resource_limits()
    return sandbox

def sandbox_plugin(plugin: Any, policy: SecurityPolicy) -> SandboxedPluginWrapper:
    """Wrap plugin in sandbox."""
    sandbox = create_sandbox(policy)
    return SandboxedPluginWrapper(plugin, sandbox)

# Example usage
if __name__ == "__main__":
    # Create policy
    policy = SecurityPolicy(
        permissions=[Permission.FILE_READ, Permission.NETWORK],
        resource_limits=ResourceLimits(
            max_memory_mb=256,
            max_cpu_percent=25,
            max_execution_time_sec=60
        ),
        allowed_hosts=["api.example.com"],
        allowed_paths=["/data/public"],
        audit_all_operations=True
    )
    
    # Create sandbox
    sandbox = create_sandbox(policy)
    
    # Test permissions
    print(f"Has file read: {sandbox.check_permission(Permission.FILE_READ)}")
    print(f"Has file write: {sandbox.check_permission(Permission.FILE_WRITE)}")
    
    # Test validation
    print(f"Can read /data/public/file.txt: {sandbox.validate_file_access('/data/public/file.txt')}")
    print(f"Can write /data/public/file.txt: {sandbox.validate_file_access('/data/public/file.txt', write=True)}")
    print(f"Can access api.example.com: {sandbox.validate_network_access('api.example.com')}")
    print(f"Can access evil.com: {sandbox.validate_network_access('evil.com')}")