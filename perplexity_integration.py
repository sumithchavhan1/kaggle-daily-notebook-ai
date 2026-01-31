#!/usr/bin/env python3
import json
import logging
from typing import Dict
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)


class PerplexityNotebookGenerator:
    """Generate notebook content using Groq AI API (OpenAI compatible)."""

    def __init__(self, api_key: str):
        """Initialize with Groq API key."""
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"
        self.timeout = 60

        # HTTP client without proxies to avoid compatibility issues
        http_client = httpx.Client()

        # OpenAI-compatible client pointed at Groq
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
        )

    def generate_notebook(
        self,
        dataset_ref: str,
        dataset_title: str,
        dataset_description: str,
        custom_prompt: str,
    ) -> str:
        """
        Generate notebook content using the custom prompt from main.py.
        This is the method main.py expects to call.
        """
        try:
            logger.info(f"Generating notebook for dataset: {dataset_title}")
            logger.info(f"Dataset ref: {dataset_ref}")
            return self.generate_notebook_content(custom_prompt)
        except Exception as e:
            logger.error(f"Error in generate_notebook: {str(e)}")
            return self._generate_template_notebook()

    def generate_notebook_content(self, prompt: str) -> str:
        """Generate notebook content from prompt using Groq."""
        try:
            logger.info("Calling Groq API for notebook generation...")

            system_prompt = (
                "You are an expert Kaggle data scientist. "
                "Return a high-quality notebook as MARKDOWN plus ```python code``` blocks. "
                "Do NOT output JSON; only headings, text, and code fences."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=4000,
            )

            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    logger.info("Successfully generated notebook content with Groq")
                    # Always treat as text (markdown + code fences)
                    return self._format_notebook_content(content)

            logger.error("Groq API returned empty content")
            return self._generate_template_notebook()

        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return self._generate_template_notebook()

    def _format_notebook_content(self, content: str) -> str:
        """Format Groq text content into a clean nbformat-4 notebook."""
        # Normalize newlines to real \n
        text = content.replace("\r\n", "\n").replace("\r", "\n")

        # Add spacing before headings so they don't stick to previous text
        text = text.replace("## ", "\n\n## ").replace("### ", "\n\n### ")

        def markdown_cell(text_block: str) -> Dict:
            block = text_block.strip()
            if not block:
                return None
            lines = block.split("\n")
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": [ln + "\n" for ln in lines],
            }

        def code_cell(code_block: str) -> Dict:
            block = code_block.strip()
            if not block:
                return None
            lines = code_block.split("\n")
            # strip optional language tag at top
            if lines and lines[0].strip().lower() in ("python", "python3"):
                lines = lines[1:]
            # expand ';' into separate lines for readability
            expanded = []
            for line in lines:
                # ignore trailing literal "\n" artifacts if any slipped through
                if line.endswith(r"\n") and not line.endswith("\\\\n"):
                    line = line[:-2]
                for part in line.split(";"):
                    part = part.rstrip()
                    if part:
                        if not part.endswith("\n"):
                            part += "\n"
                        expanded.append(part)
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": expanded,
            }

        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        # Split by ``` fences into markdown / code blocks
        blocks = text.split("```")
        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            if i % 2 == 0:
                cell = markdown_cell(block)
            else:
                cell = code_cell(block)
            if cell:
                notebook["cells"].append(cell)

        return json.dumps(notebook, indent=2)

    def _generate_template_notebook(self) -> str:
        """Generate template notebook as fallback."""
        logger.info("Generating template notebook")
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Kaggle Notebook: ML Analysis\n"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import pandas as pd\n", "import numpy as np\n"],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }
        return json.dumps(notebook, indent=2)
