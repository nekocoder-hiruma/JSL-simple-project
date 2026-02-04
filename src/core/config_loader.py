import configparser
import os
from typing import List, Any

class ConfigLoader:
    """
    Handles loading and retrieving configuration settings from .ini files.
    Allows for a base configuration to be layered over with model-specific settings.
    """
    def __init__(self, config_files: List[str]):
        self.config = configparser.ConfigParser()
        self.config.read(config_files)

    def get_setting(self, section: str, option: str, fallback: Any = None) -> str:
        """Retrieves a text configuration setting."""
        return self.config.get(section, option, fallback=fallback)

    def get_int(self, section: str, option: str, fallback: Any = None) -> int:
        """Retrieves an integer configuration setting."""
        return self.config.getint(section, option, fallback=fallback)

    def get_float(self, section: str, option: str, fallback: Any = None) -> float:
        """Retrieves a decimal configuration setting."""
        return self.config.getfloat(section, option, fallback=fallback)

def load_system_config() -> ConfigLoader:
    """Loads the general system settings for the project."""
    # Get the directory of the current file (src/core/)
    core_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to src/ and then into configs/
    path = os.path.join(core_dir, '..', 'configs', 'base_config.ini')
    return ConfigLoader([path])

def load_specialized_config(model_type: str) -> ConfigLoader:
    """Loads system settings and overrides them with specific model requirements."""
    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(core_dir, '..', 'configs', 'base_config.ini')
    model_path = os.path.join(core_dir, '..', 'configs', f'{model_type}_config.ini')
    return ConfigLoader([base_path, model_path])
