import sys
import importlib
from pathlib import Path

def reload_project():
    """Reload all project modules."""
    project_root = Path(__file__).parent
    src_dir = project_root / 'src'
    
    # Get all Python files in the project
    python_files = list(src_dir.rglob('*.py'))
    
    # Convert file paths to module names
    modules = []
    for file in python_files:
        if file.stem == '__init__':
            continue
        
        relative_path = file.relative_to(project_root)
        module_path = str(relative_path.with_suffix('')).replace('/', '.')
        
        try:
            module = sys.modules.get(module_path)
            if module:
                modules.append(module)
        except Exception:
            continue
    
    # Reload modules in reverse order to handle dependencies
    for module in reversed(modules):
        try:
            importlib.reload(module)
        except Exception:
            continue

    return len(modules) 