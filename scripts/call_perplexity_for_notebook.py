"""Call Perplexity API to generate comprehensive notebook cells."""
import json
import os
from typing import Any, Dict, List, Optional

import requests


PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "pplx-7b-online"


class PerplexityNotebookClient:
    """
    Thin client around Perplexity Chat Completions API to generate
    Jupyter notebook cells for a given Kaggle dataset.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY is not set.")
        self.model = model
        self.timeout = timeout

    def _build_system_prompt(self) -> str:
        """System prompt: how the assistant should behave and format output."""
        return (
            "You are a senior Kaggle Grandmaster data scientist.\n"
            "Generate a COMPLETE Jupyter notebook for Kaggle in JSON form, as a list of cells.\n\n"
            "Constraints:\n"
            "- Output ONLY valid JSON (no backticks, no explanation).\n"
            "- JSON must be a list of cell objects.\n"
            "- Each cell object must have: 'cell_type', 'metadata', 'source' (list of strings).\n"
            "- Do NOT include 'execution_count' or 'outputs'.\n\n"
            "Notebook style:\n"
            "- Professional, production-ready analysis for Kaggle.\n"
            "- Clear sections: title, EDA, visualizations, feature engineering, modeling, evaluation, conclusion.\n"
            "- Use idiomatic pandas, numpy, matplotlib, seaborn, sklearn.\n"
            "- Add comments in code cells explaining key steps.\n"
            "- Avoid unnecessary prints; focus on useful outputs.\n"
        )

    def _build_user_prompt(self, dataset_meta: Dict[str, Any]) -> str:
        """User prompt: describe this particular dataset and desired workflow."""
        title = dataset_meta.get("title", "Kaggle Dataset")
        slug = dataset_meta.get("dataset_slug", "")
        file_path = dataset_meta.get("file_path", "kaggle/input/dataset/file.csv")
        target = dataset_meta.get("target_column", "")
        task = dataset_meta.get("task_type", "auto")
        description = dataset_meta.get("description", "")

        lines = [
            f"Title: {title}\n",
            f"Dataset slug: {slug}\n",
            f"CSV path (for Kaggle): {file_path}\n",
            f"Target column: {target}\n",
            f"Task type: {task}\n",
            "\n",
            "Goal:\n",
            "- Generate a complete Kaggle notebook that loads this dataset, performs strong EDA and builds at least one baseline ML model.\n",
            "- Match the style of polished Kaggle notebooks with:\n",
            "  * Markdown title and description.\n",
            "  * Import cell with numpy, pandas, matplotlib, seaborn, sklearn, scipy if needed.\n",
            "  * Data loading cell using the given CSV path.\n",
            "  * Data overview (head, info, describe, missing values, duplicates).\n",
            "  * Visualizations: distributions, correlations, time-series or category plots as appropriate.\n",
            "  * Feature engineering and encoding where needed.\n",
            "  * Train/test split.\n",
            "  * At least one model (e.g., LinearRegression / RandomForest / GradientBoosting for regression or classifiers for classification).\n",
            "  * Metrics and model comparison (R2/RMSE/MAE for regression, accuracy/F1/ROC-AUC for classification).\n",
            "  * Feature importance plots if model supports it.\n",
            "  * Final conclusion markdown summarizing key insights.\n",
            "\n",
            "Technical requirements:\n",
            "- Use a single main DataFrame named 'df'.\n",
            "- Use seaborn and matplotlib for all plots.\n",
            "- Use 'train_test_split' from sklearn.model_selection.\n",
            "- Use clear variable names and comments.\n",
            "- Do NOT include any magic commands except '%matplotlib inline' if needed.\n",
            "- Assume the notebook runs on Kaggle with the dataset available at the given path.\n",
            "\n",
            "Output format:\n",
            "- ONLY output JSON (no backticks, no markdown around it).\n",
            "- JSON is a list of cell dicts with 'cell_type', 'metadata', and 'source' as list of lines.\n",
        ]
        if description:
            lines.insert(1, f"Short dataset description: {description}\n")
        return "".join(lines)

    def call_api(self, dataset_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call Perplexity and return parsed cell list."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(dataset_meta)},
            ],
            "temperature": 0.4,
            "max_tokens": 4096,
            "top_p": 0.9,
        }

        resp = requests.post(
            PERPLEXITY_API_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Perplexity API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected API response format: {data}") from exc

        # content should be JSON string (list of cells)
        try:
            cells = json.loads(content)
        except json.JSONDecodeError as exc:
            # Optional: try to salvage by stripping backticks if model added them
            text = content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if "\n" in text:
                    text = "\n".join(text.split("\n")[1:])
            try:
                cells = json.loads(text)
            except json.JSONDecodeError as exc2:
                raise RuntimeError(
                    f"Failed to parse notebook JSON from model output: {exc2}\n\nRaw content:\n{content[:2000]}"
                ) from exc

        if not isinstance(cells, list):
            raise RuntimeError("Model output is not a list of cells.")

        return cells


def generate_notebook_cells(dataset_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Public function used by the orchestrator."""
    client = PerplexityNotebookClient()
    return client.call_api(dataset_meta)


if __name__ == "__main__":
    # Simple local test
    example_meta = {
        "title": "Sample Dataset",
        "dataset_slug": "user/sample-dataset",
        "file_path": "kaggle/input/sample-dataset/file.csv",
        "target_column": "target",
        "task_type": "classification",
        "description": "Sample dataset for testing.",
    }

    cells = generate_notebook_cells(example_meta)
    print(f"Generated {len(cells)} cells")
    print(json.dumps(cells[:2], indent=2))  # Print first 2 cells
