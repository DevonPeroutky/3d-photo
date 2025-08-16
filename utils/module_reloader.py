"""Module reloader utility for Grasshopper development workflow.

This utility provides automatic reloading of custom Python modules to ensure
Grasshopper picks up changes without requiring a Rhino restart.
"""

import importlib
import sys
from typing import List, Optional


class ModuleReloader:
    """Handles automatic reloading of custom Python modules for development."""
    
    def __init__(self, debug: bool = False):
        """Initialize the module reloader.
        
        Args:
            debug: If True, print debug information about reload operations
        """
        self.debug = debug
        self.reload_count = 0
    
    def reload_custom_modules(self, module_names: Optional[List[str]] = None) -> bool:
        """Reload custom modules in dependency order.
        
        Args:
            module_names: List of module names to reload. If None, uses default set.
            
        Returns:
            True if all modules reloaded successfully, False otherwise
        """
        if module_names is None:
            # Default modules in dependency order
            module_names = [
                'custom_types',
                'performance_monitor', 
                'color_utils',
                'gaussian_splat_reader'
            ]
        
        success = True
        reloaded_modules = []
        
        for module_name in module_names:
            if self._reload_module(module_name):
                reloaded_modules.append(module_name)
            else:
                success = False
                
        self.reload_count += 1
        
        if self.debug:
            print(f"Reload #{self.reload_count} - Reloaded {len(reloaded_modules)} modules: {reloaded_modules}")
            if not success:
                print("Some modules failed to reload - check for import errors")
                
        return success
    
    def _reload_module(self, module_name: str) -> bool:
        """Reload a single module if it exists in sys.modules.
        
        Args:
            module_name: Name of the module to reload
            
        Returns:
            True if module was reloaded successfully, False otherwise
        """
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                if self.debug:
                    print(f"  ✓ Reloaded: {module_name}")
                return True
            else:
                if self.debug:
                    print(f"  ⚠ Module not loaded yet: {module_name}")
                return True  # Not an error if module hasn't been imported yet
                
        except Exception as e:
            if self.debug:
                print(f"  ✗ Failed to reload {module_name}: {e}")
            return False


# Global instance for easy use in scripts
reloader = ModuleReloader(debug=False)


def reload_all_custom_modules(debug: bool = False) -> bool:
    """Convenience function to reload all custom modules.
    
    Args:
        debug: If True, print debug information about reload operations
        
    Returns:
        True if all modules reloaded successfully, False otherwise
    """
    global reloader
    reloader.debug = debug
    return reloader.reload_custom_modules()