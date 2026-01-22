import subprocess
import json
from datetime import datetime

def fetch_trending_dataset():
    """Fetch trending dataset from Kaggle using CLI"""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--sort-by", "hottest", "--max-size", "1000000000"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Kaggle API error: {result.stderr}")
        
        lines = result.stdout.strip().split('\n')[1:]
        if not lines:
            raise Exception("No trending datasets found")
        
        dataset_ref = lines[0].split()[0]
        owner, slug = dataset_ref.split('/')
        
        try:
            meta_result = subprocess.run(
                ["kaggle", "datasets", "metadata", dataset_ref],
                capture_output=True,
                text=True
            )
            if meta_result.returncode == 0:
                metadata = json.loads(meta_result.stdout)
                dataset_title = metadata.get('title', slug)
            else:
                dataset_title = slug.replace('-', ' ').title()
        except:
            dataset_title = slug.replace('-', ' ').title()
        
        dataset_data = {
            'dataset_slug': dataset_ref,
            'owner': owner,
            'slug': slug,
            'dataset_title': dataset_title,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        print(f"✓ Dataset: {dataset_ref} - {dataset_title}")
        return dataset_data
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        raise
