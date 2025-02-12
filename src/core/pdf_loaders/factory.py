from enum import Enum
from typing import Type, Optional
import importlib
import logging
from .base_loader import BasePDFLoader

logger = logging.getLogger(__name__)

class PDFLoaderType(Enum):
    FITZ = "fitz"
    DOCLING = "docling"

class PDFLoaderFactory:
    """Factory for creating PDF loaders with lazy loading of dependencies."""
    
    _loader_instance: Optional[BasePDFLoader] = None
    
    @classmethod
    def create(cls, loader_type: str) -> BasePDFLoader:
        """
        Create a PDF loader instance based on the specified type.
        Lazily imports required dependencies.
        """
        if cls._loader_instance:
            return cls._loader_instance
            
        try:
            loader_enum = PDFLoaderType(loader_type.lower())
            
            if loader_enum == PDFLoaderType.FITZ:
                # Lazy import fitz loader
                FitzLoader = cls._import_fitz_loader()
                cls._loader_instance = FitzLoader()
            elif loader_enum == PDFLoaderType.DOCLING:
                # Lazy import docling loader
                DoclingLoader = cls._import_docling_loader()
                cls._loader_instance = DoclingLoader()
                
            return cls._loader_instance
            
        except (KeyError, ValueError) as e:
            raise ValueError(f"Unsupported PDF loader type: {loader_type}") from e
        except ImportError as e:
            raise ImportError(f"Required dependencies for {loader_type} loader not installed: {str(e)}")
    
    @staticmethod
    def _import_fitz_loader() -> Type[BasePDFLoader]:
        """Lazily import and return the Fitz loader class."""
        try:
            import fitz  # Only import when needed
            from .fitz_loader import FitzPDFLoader
            logger.info("Successfully loaded PyMuPDF (fitz) loader")
            return FitzPDFLoader
        except ImportError as e:
            logger.error("Failed to import PyMuPDF (fitz) loader: %s", str(e))
            raise ImportError("PyMuPDF (fitz) is required for this loader") from e
    
    @staticmethod
    def _import_docling_loader() -> Type[BasePDFLoader]:
        """Lazily import and return the Docling loader class."""
        try:
            from docling.document_converter import DocumentConverter  # Only import when needed
            from .docling_loader import DoclingPDFLoader
            logger.info("Successfully loaded Docling loader")
            return DoclingPDFLoader
        except ImportError as e:
            logger.error("Failed to import Docling loader: %s", str(e))
            raise ImportError("Docling is required for this loader") from e 