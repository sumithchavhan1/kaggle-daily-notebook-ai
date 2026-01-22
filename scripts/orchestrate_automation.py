"""Main orchestration script for daily Kaggle notebook generation"""
import sys
from pathlib import Path

try:
    from fetch_trending_dataset import fetch_trending_dataset
    from call_perplexity_for_notebook import call_perplexity_for_notebook
    from build_notebook import build_notebook_from_spec
    from publish_to_kaggle import publish_notebook_to_kaggle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    try:
        print("[STEP 1/4] Fetching trending dataset from Kaggle...")
        dataset_meta = fetch_trending_dataset()
        print(f"✓ Dataset selected: {dataset_meta['dataset_slug']}")
        
        print("\n[STEP 2/4] Calling Perplexity API to generate notebook...")
        notebook_spec = call_perplexity_for_notebook(dataset_meta)
        print(f"✓ Generated notebook with {len(notebook_spec.get('cells', []))} cells")
        
        print("\n[STEP 3/4] Building Jupyter notebook file...")
        notebook_info = build_notebook_from_spec(notebook_spec, dataset_meta)
        print(f"✓ Notebook saved to: {notebook_info['notebook_path']}")
        
        print("\n[STEP 4/4] Publishing notebook to Kaggle...")
        publish_notebook_to_kaggle(notebook_info, dataset_meta)
        print("✓ Notebook published to Kaggle!")
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
