import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

def publish_notebook_to_kaggle(notebook_info, dataset_meta):
    """Publish notebook to Kaggle using Kaggle API"""
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Create kernel metadata
        kernel_metadata = {
            "id": f"sumitchavhan7/daily-trending-{date_str}",
            "title": f"Daily Trending Notebook – {date_str}",
            "code_file": notebook_info['notebook_path'],
            "language": "python",
            "kernel_type": "notebook",
            "is_private": False,
            "enable_gpu": False,
            "enable_tpu": False,
            "dataset_sources": [dataset_meta['dataset_slug']],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        # Create .kaggle directory and metadata file
        kaggle_dir = Path(".kaggle")
        kaggle_dir.mkdir(exist_ok=True)
        
        metadata_path = kaggle_dir / "kernel-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(kernel_metadata, f, indent=2)
        
        # Push notebook to Kaggle
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", "."],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ Kernel published successfully!")
            print(f"✓ Title: {kernel_metadata['title']}")
        else:
            print(f"⚠ Kaggle API response: {result.stdout}")
            print(f"⚠ Note: Kernel push may require manual verification")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise
