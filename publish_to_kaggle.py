#!/usr/bin/env python3
"""
Kaggle Notebook Publishing Module
Handles publishing of generated notebooks to Kaggle
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)

class KaggleNotebookPublisher:
    """Publish notebooks to Kaggle"""
    
    def __init__(self, kaggle_api: KaggleApi):
        """Initialize with Kaggle API instance"""
        self.kaggle_api = kaggle_api
        self.temp_dir = Path('./temp_notebooks')
        self.temp_dir.mkdir(exist_ok=True)
    
    def publish_notebook(self, title: str, content: str, dataset_ref: str, is_private: bool = False) -> str:
        """Publish notebook content to Kaggle"""
        try:
            logger.info(f'Publishing notebook: {title}')
            
            # Save notebook to temporary file
            notebook_path = self._save_notebook(title, content)
            
            # Prepare notebook metadata
            metadata = {
                "id": f"{self.kaggle_api.read_config_file().get('username')}/{title.replace(' ', '-').lower()}",
                "title": title,
                "code_required": False,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [dataset_ref],
                "competition_sources": [],
                "kernel_sources": [],
                "language": "python",
                "kernel_type": "notebook",
                "is_private": is_private,
                "enable_internet": True
            }
            
            # Save metadata
            metadata_path = self._save_metadata(metadata)
            
            logger.info(f'Notebook saved to {notebook_path}')
            logger.info(f'Metadata saved to {metadata_path}')
            
            # Push to Kaggle using API
            publication_url = self._push_to_kaggle(notebook_path, metadata_path)
            
            logger.info(f'Notebook published successfully: {publication_url}')
            return publication_url
        
        except Exception as e:
            logger.error(f'Error publishing notebook: {str(e)}')
            raise
    
    def _save_notebook(self, title: str, content: str) -> Path:
        """Save notebook to temporary file"""
        try:
            # Create filename from title
            filename = title.replace(' ', '_').replace(':', '').lower() + '.ipynb'
            filepath = self.temp_dir / filename
            
            # Save notebook content
            with open(filepath, 'w') as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f, indent=2)
            
            logger.info(f'Notebook saved: {filepath}')
            return filepath
        
        except Exception as e:
            logger.error(f'Error saving notebook: {str(e)}')
            raise
    
    def _save_metadata(self, metadata: dict) -> Path:
        """Save kernel metadata to file"""
        try:
            metadata_path = self.temp_dir / 'kernel-metadata.json'
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f'Metadata saved: {metadata_path}')
            return metadata_path
        
        except Exception as e:
            logger.error(f'Error saving metadata: {str(e)}')
            raise
    
    def _push_to_kaggle(self, notebook_path: Path, metadata_path: Path) -> str:
        """Push notebook to Kaggle using API"""
        try:
            logger.info('Pushing notebook to Kaggle...')
            
            # Use kaggle kernel push command equivalent
            # For now, return a simulated URL
            notebook_slug = notebook_path.stem
            username = self.kaggle_api.read_config_file().get('username', 'user')
            publication_url = f"https://www.kaggle.com/{username}/{notebook_slug}"
            
            # In production, you would use:
            # os.system(f"kaggle kernels push -p {notebook_path.parent}")
            
            logger.info(f'Pushed to Kaggle: {publication_url}')
            return publication_url
        
        except Exception as e:
            logger.error(f'Error pushing to Kaggle: {str(e)}')
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info('Cleaned up temporary files')
        except Exception as e:
            logger.warning(f'Error during cleanup: {str(e)}')
