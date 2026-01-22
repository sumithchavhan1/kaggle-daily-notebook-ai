import json
from pathlib import Path
from datetime import datetime
import nbformat as nbf

def build_notebook_from_spec(notebook_spec, dataset_meta):
    """Convert notebook spec JSON into a valid .ipynb file"""
    
    # Create notebook
    nb = nbf.v4.new_notebook()
    
    # Add metadata
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }
    
    # Add cells from spec
    for cell_spec in notebook_spec.get('cells', []):
        if cell_spec['type'] == 'markdown':
            nb.cells.append(nbf.v4.new_markdown_cell(cell_spec['content']))
        elif cell_spec['type'] == 'code':
            nb.cells.append(nbf.v4.new_code_cell(cell_spec['content']))
    
    # Generate filename with date
    date_str = datetime.now().strftime('%Y-%m-%d')
    notebook_filename = f"notebook-{date_str}.ipynb"
    
    notebook_path = Path("trending_notebooks") / notebook_filename
    notebook_path.parent.mkdir(exist_ok=True)
    
    # Write notebook
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"✓ Notebook saved: {notebook_path}")
    
    return {
        'notebook_path': str(notebook_path),
        'notebook_filename': notebook_filename,
        'notebook_title': notebook_spec.get('notebook_title', 'Generated Notebook'),
        'notebook_slug': notebook_spec.get('notebook_slug', f'daily-notebook-{date_str}')
    }
