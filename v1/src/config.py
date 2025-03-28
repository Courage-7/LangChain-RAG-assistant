import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager for the RAG Document Assistant"""
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern to ensure only one config instance exists"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join("config", "config.yaml")
        
        try:
            with open(config_path, 'r') as file:
                self._config = yaml.safe_load(file)
                
                # Set default vector store to FAISS if not specified
                if 'vector_store' in self._config and 'type' not in self._config['vector_store']:
                    self._config['vector_store']['type'] = 'faiss'
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self._config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with dot notation support"""
        if not self._config:
            return default
            
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config.copy() if self._config else {}
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment variable"""
        return os.environ.get("OPENAI_API_KEY", "")