from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePDFLoader(ABC):
    """Base interface for PDF loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load and process a PDF file."""
        pass 