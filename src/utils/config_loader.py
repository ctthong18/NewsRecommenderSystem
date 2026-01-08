import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from copy import deepcopy


class ConfigLoader:
    """
    Load and manage configuration from YAML files with support for:
    - Nested configuration structures
    - Environment variable overrides
    - Configuration validation
    - Default values
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigLoader.
        
        Args:
            config_path: Path to YAML config file. If None, uses empty config.
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        
        if self.config_path and self.config_path.exists():
            self.load()
    
    def load(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file. If None, uses self.config_path.
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        return self.config
    
    def _apply_env_overrides(self):
        """
        Apply environment variable overrides to config.
        
        Environment variables should be in format: CONFIG_SECTION_KEY=value
        Example: CONFIG_MODEL_LR=0.001 overrides config['model']['lr']
        """
        prefix = "CONFIG_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Remove prefix and split by underscore
            config_path = env_key[len(prefix):].lower().split('_')
            
            # Navigate to the nested config location
            current = self.config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value with type conversion
            final_key = config_path[-1]
            current[final_key] = self._convert_type(env_value)
    
    def _convert_type(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value (int, float, bool, or str)
        """
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.lr')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('model.lr', 0.001)
            0.0001
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.lr')
            value: Value to set
            
        Example:
            >>> config.set('model.lr', 0.001)
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against a schema.
        
        Args:
            schema: Dictionary defining required keys and their types
                   Format: {'key.path': type} or {'key.path': (type, default)}
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
            
        Example:
            >>> schema = {
            ...     'model.lr': float,
            ...     'model.batch_size': int,
            ...     'data.train_path': str
            ... }
            >>> config.validate(schema)
        """
        errors = []
        
        for key_path, expected in schema.items():
            # Handle (type, default) tuple format
            if isinstance(expected, tuple):
                expected_type, default_value = expected
                value = self.get(key_path, default_value)
                # Set default if missing
                if self.get(key_path) is None:
                    self.set(key_path, default_value)
            else:
                expected_type = expected
                value = self.get(key_path)
            
            # Check if required key exists
            if value is None:
                errors.append(f"Missing required config key: {key_path}")
                continue
            
            # Check type
            if not isinstance(value, expected_type):
                errors.append(
                    f"Invalid type for {key_path}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def merge(self, other_config: Dict[str, Any]):
        """
        Merge another configuration dictionary into current config.
        Nested dictionaries are merged recursively.
        
        Args:
            other_config: Configuration dictionary to merge
        """
        self.config = self._deep_merge(self.config, other_config)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary with updates
            
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return deepcopy(self.config)
    
    def save(self, output_path: Union[str, Path]):
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save config file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)


def load_config(
    config_path: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> ConfigLoader:
    """
    Convenience function to load and validate configuration.
    
    Args:
        config_path: Path to YAML config file
        schema: Optional validation schema
        overrides: Optional dictionary to override config values
        
    Returns:
        ConfigLoader instance with loaded configuration
        
    Example:
        >>> config = load_config(
        ...     'configs/train.yaml',
        ...     schema={'model.lr': float, 'model.batch_size': int}
        ... )
    """
    loader = ConfigLoader(config_path)
    
    if overrides:
        loader.merge(overrides)
    
    if schema:
        loader.validate(schema)
    
    return loader
