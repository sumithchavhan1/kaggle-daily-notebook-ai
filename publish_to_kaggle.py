#!/usr/bin/env python3
"""
Kaggle Notebook Publishing Module
Handles publishing of generated notebooks to Kaggle
"""
import os
import json
import logging
import subprocess
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
        """Publish notebook content to Kaggle with automatic version handling for duplicate titles"""
        try:
            logger.info(f'Publishing notebook: {title}')
            
            # Try to publish with the original title, and if it fails due to conflict, add version suffix
            original_title = title
            version = 1
            current_title = title
            max_attempts = 10  # Try up to V10
            
            while version <= max_attempts:
                try:
                    # Save notebook to temporary file
                    notebook_path = self._save_notebook(current_title, content)
                    
                    # Prepare notebook metadata
                    metadata = {
                        "id": f"{self.kaggle_api.read_config_file().get('username')}/notebook-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "title": current_title,
                        "code_required": False,
                        "enable_gpu": False,
                        "enable_internet": True,
                        "dataset_sources": [dataset_ref],
                        "competition_sources": [],
                        "kernel_sources": [],
                        "language": "python",
                        "kernel_type": "notebook",
                        "is_private": is_private,
                        "enable_internet": True,
                        "code_file": notebook_path.name
                    }
                    
                    # Save metadata
                    metadata_path = self._save_metadata(metadata)
                    
                    logger.info(f'Notebook saved to {notebook_path}')
                    logger.info(f'Metadata saved to {metadata_path}')
                    
                    # Push to Kaggle using API
                    publication_url = self._push_to_kaggle(notebook_path, metadata_path)
                    
                    logger.info(f'Notebook published successfully: {publication_url}')
                    if current_title != original_title:
                        logger.info(f'Published with modified title: {current_title} (original: {original_title})')
                    return publication_url
                    
                except Exception as e:
                    error_str = str(e)
                    # Check if error is due to title conflict (409 error or "already in use" message)
                    if ('409' in error_str or 'already in use' in error_str or 'Conflict' in error_str) and version < max_attempts:
                        version += 1
                        current_title = f"{original_title} - V{version}"
                        logger.warning(f'Title conflict detected. Retrying with title: {current_title}')
                        # Clean up the failed attempt
                        try:
                            notebook_path_failed = self.temp_dir / f"{original_title.replace(' ', '_').replace(':', '').lower()}.ipynb"
                            if notebook_path_failed.exists():
                                notebook_path_failed.unlink()
                        except:
                            pass
                        continue
                    else:
                        # Not a conflict error or max attempts reached
                        logger.error(f'Error publishing notebook: {error_str}')
                        raise
            
            # If we get here, we couldn't publish after max attempts
            raise Exception(f'Failed to publish notebook after {max_attempts} attempts with different titles')
            
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
            
            # Use kaggle kernel push command with subprocess to capture output
            notebook_slug = notebook_path.stem
            username = self.kaggle_api.read_config_file().get('username', 'user')
            
            # Execute kaggle kernels push command using subprocess
            kernel_dir = notebook_path.parent
            cmd = ["kaggle", "kernels", "push"]
            
            logger.info(f'Executing: {" ".join(cmd)} in {kernel_dir}')
            
            # Use subprocess with PIPE to capture both stdout and stderr
            process = subprocess.Popen(
                cmd,
                cwd=str(kernel_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            # Log the output for debugging
            if stdout:
                logger.info(f'Kaggle stdout: {stdout}')
            if stderr:
                logger.info(f'Kaggle stderr: {stderr}')
            
            # Check for conflict error in stderr or stdout
            combined_output = stdout + stderr
            if '409' in combined_output or 'already in use' in combined_output or 'Conflict' in combined_output:
                raise Exception(f'Title conflict error detected: {combined_output}')
            
            if process.returncode == 0:
                publication_url = f"https://www.kaggle.com/{username}/{notebook_slug}"
                logger.info(f'Kernel push successful: {publication_url}')
                return publication_url
            else:
                error_msg = f'Kernel push failed with code {process.returncode}. Stderr: {stderr}'
                raise Exception(error_msg)
            
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
