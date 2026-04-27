"""
Plugin Lifecycle Manager

Manage plugin init, config, health, shutdown.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import threading
import time
from datetime import datetime

from .plugin_interface import (
    BasePlugin, PluginRegistry, PluginStatus, PluginType, PluginMetadata
)

logger = logging.getLogger(__name__)

@dataclass
class PluginConfig:
    """Plugin config"""
    name: str
    plugin_class: str
    config: Dict[str, Any]
    auto_start: bool = True
    health_check_interval: int = 60  # seconds
    restart_on_failure: bool = True
    max_restart_attempts: int = 3

class PluginManager:
    """Plugin lifecycle manager"""
    
    def __init__(self):
        self.registry = PluginRegistry()
        self.configs: Dict[str, PluginConfig] = {}
        self.health_check_threads: Dict[str, threading.Thread] = {}
        self.restart_attempts: Dict[str, int] = {}
        self.running = False
    
    def load_plugin(self, config: PluginConfig, plugin_instance: BasePlugin) -> bool:
        """Load + init plugin."""
        try:
            # Register
            if not self.registry.register(config.name, plugin_instance):
                logger.error(f"Plugin {config.name} already registered")
                return False
            
            # Store config
            self.configs[config.name] = config
            self.restart_attempts[config.name] = 0
            
            # Init if auto_start
            if config.auto_start:
                if not self._initialize_plugin(config.name):
                    return False
            
            # Start health check
            if config.health_check_interval > 0:
                self._start_health_check(config.name)
            
            logger.info(f"Plugin {config.name} loaded")
            return True
            
        except Exception as e:
            logger.error(f"Load plugin {config.name} fail: {e}")
            return False
    
    def unload_plugin(self, name: str) -> bool:
        """Unload plugin."""
        try:
            # Stop health check
            self._stop_health_check(name)
            
            # Shutdown plugin
            plugin = self.registry.get(name)
            if plugin:
                plugin.shutdown()
            
            # Unregister
            self.registry.unregister(name)
            
            # Cleanup
            if name in self.configs:
                del self.configs[name]
            if name in self.restart_attempts:
                del self.restart_attempts[name]
            
            logger.info(f"Plugin {name} unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Unload plugin {name} fail: {e}")
            return False
    
    def _initialize_plugin(self, name: str) -> bool:
        """Init plugin."""
        plugin = self.registry.get(name)
        if not plugin:
            return False
        
        try:
            success = plugin.initialize()
            if success:
                plugin.status = PluginStatus.ACTIVE
                logger.info(f"Plugin {name} initialized")
            else:
                plugin.status = PluginStatus.ERROR
                logger.error(f"Plugin {name} init fail")
            return success
            
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            logger.error(f"Plugin {name} init exception: {e}")
            return False
    
    def _start_health_check(self, name: str):
        """Start health check thread."""
        if name in self.health_check_threads:
            return
        
        thread = threading.Thread(
            target=self._health_check_loop,
            args=(name,),
            daemon=True
        )
        thread.start()
        self.health_check_threads[name] = thread
    
    def _stop_health_check(self, name: str):
        """Stop health check thread."""
        if name in self.health_check_threads:
            # Thread will exit when plugin unregistered
            del self.health_check_threads[name]
    
    def _health_check_loop(self, name: str):
        """Health check loop."""
        config = self.configs.get(name)
        if not config:
            return
        
        while name in self.registry.plugins:
            try:
                plugin = self.registry.get(name)
                if not plugin:
                    break
                
                # Health check
                health = plugin.health_check()
                
                if not health.get('healthy', False):
                    logger.warning(f"Plugin {name} unhealthy: {health}")
                    
                    # Restart if enabled
                    if config.restart_on_failure:
                        self._attempt_restart(name)
                
                # Sleep
                time.sleep(config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check {name} fail: {e}")
                time.sleep(config.health_check_interval)
    
    def _attempt_restart(self, name: str):
        """Attempt plugin restart."""
        config = self.configs.get(name)
        if not config:
            return
        
        attempts = self.restart_attempts.get(name, 0)
        
        if attempts >= config.max_restart_attempts:
            logger.error(f"Plugin {name} max restart attempts reached")
            return
        
        logger.info(f"Restarting plugin {name} (attempt {attempts + 1})")
        
        plugin = self.registry.get(name)
        if plugin:
            # Shutdown
            try:
                plugin.shutdown()
            except Exception as e:
                logger.error(f"Shutdown {name} fail: {e}")
            
            # Reinit
            if self._initialize_plugin(name):
                self.restart_attempts[name] = 0
                logger.info(f"Plugin {name} restarted")
            else:
                self.restart_attempts[name] = attempts + 1
                logger.error(f"Plugin {name} restart fail")
    
    def get_plugin_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin status."""
        plugin = self.registry.get(name)
        if not plugin:
            return None
        
        metadata = plugin.get_metadata()
        health = plugin.health_check()
        
        return {
            'name': name,
            'status': plugin.get_status().value,
            'type': metadata.plugin_type.value,
            'version': metadata.version,
            'vendor': metadata.vendor,
            'health': health,
            'restart_attempts': self.restart_attempts.get(name, 0)
        }
    
    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get all plugin status."""
        return [
            self.get_plugin_status(name)
            for name in self.registry.list_plugins()
        ]
    
    def start_all(self) -> bool:
        """Start all plugins."""
        self.running = True
        success = True
        
        for name in self.registry.list_plugins():
            plugin = self.registry.get(name)
            if plugin and plugin.get_status() != PluginStatus.ACTIVE:
                if not self._initialize_plugin(name):
                    success = False
        
        return success
    
    def stop_all(self) -> bool:
        """Stop all plugins."""
        self.running = False
        success = True
        
        for name in self.registry.list_plugins():
            plugin = self.registry.get(name)
            if plugin:
                try:
                    plugin.shutdown()
                except Exception as e:
                    logger.error(f"Shutdown {name} fail: {e}")
                    success = False
        
        return success
    
    def reload_plugin(self, name: str) -> bool:
        """Reload plugin."""
        plugin = self.registry.get(name)
        if not plugin:
            return False
        
        # Shutdown
        try:
            plugin.shutdown()
        except Exception as e:
            logger.error(f"Shutdown {name} fail: {e}")
        
        # Reinit
        return self._initialize_plugin(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """Get plugin names by type."""
        plugins = self.registry.get_by_type(plugin_type)
        return [p.get_metadata().name for p in plugins]

# Example usage
if __name__ == "__main__":
    from .plugin_interface import ScannerPlugin, PluginMetadata
    
    # Mock scanner plugin
    class MockScanner(ScannerPlugin):
        def initialize(self):
            return True
        
        def shutdown(self):
            return True
        
        def health_check(self):
            return {'healthy': True}
        
        def get_metadata(self):
            return PluginMetadata(
                name="mock_scanner",
                version="1.0.0",
                plugin_type=PluginType.SCANNER,
                vendor="Test",
                description="Mock scanner",
                capabilities=["read_region"],
                config_schema={}
            )
        
        def connect(self):
            return True
        
        def disconnect(self):
            return True
        
        def get_slide_list(self):
            return ["slide1", "slide2"]
        
        def get_slide_metadata(self, slide_id):
            return None
        
        def read_region(self, slide_id, x, y, width, height, level=0):
            return None
        
        def get_thumbnail(self, slide_id, max_size=512):
            return None
    
    # Test manager
    manager = PluginManager()
    
    config = PluginConfig(
        name="test_scanner",
        plugin_class="MockScanner",
        config={},
        auto_start=True,
        health_check_interval=10
    )
    
    plugin = MockScanner({})
    
    if manager.load_plugin(config, plugin):
        print("Plugin loaded")
        
        status = manager.get_plugin_status("test_scanner")
        print(f"Status: {status}")
        
        manager.unload_plugin("test_scanner")
        print("Plugin unloaded")