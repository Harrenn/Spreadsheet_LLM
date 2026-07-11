#!/usr/bin/env python3
"""Normalize notebook configuration and remove execution state before sharing."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


REPLACEMENTS = {
    "branch_cleaner_colab.ipynb": {
        "# Edit this cell first, then run the helper/init/action cells below.": "# Edit environment variables or this cell first, then run the helper/init/action cells below.\nimport os",
        "INPUT_PATH = '/content/all_branch_data.csv'": "INPUT_PATH = os.getenv('SPREADSHEET_LLM_INPUT_PATH', '/content/all_branch_data.csv')",
        "GOOGLE_GEOCODING_API_KEY = ''": "GOOGLE_GEOCODING_API_KEY = os.getenv('GOOGLE_GEOCODING_API_KEY', '')",
        "GGUF_MODEL_PATH = '/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/02749a7b272109255a4c559a80894d3d9777574c/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'": "GGUF_MODEL_PATH = os.getenv('GGUF_MODEL_PATH', '')",
    },
    "branch_cleaner_vast_ai.ipynb": {
        "HF_TOKEN = ''": "import os\nHF_TOKEN = os.getenv('HF_TOKEN', '')",
        "# Edit this cell first, then run the helper/action cells below.": "# Edit environment variables or this cell first, then run the helper/action cells below.\nimport os",
        "INPUT_PATH = '/workspace/all_branch_data.csv'": "INPUT_PATH = os.getenv('SPREADSHEET_LLM_INPUT_PATH', '/workspace/all_branch_data.csv')",
        "GOOGLE_GEOCODING_API_KEY = ''": "GOOGLE_GEOCODING_API_KEY = os.getenv('GOOGLE_GEOCODING_API_KEY', '')",
        "GGUF_MODEL_PATH = globals().get('GGUF_MODEL_PATH', '')": "GGUF_MODEL_PATH = os.getenv('GGUF_MODEL_PATH', globals().get('GGUF_MODEL_PATH', ''))",
    },
    "occupation_standardization_cuda.ipynb": {
        "from pathlib import Path": "from pathlib import Path\nimport os",
        "INPUT_PATH = find_single_input_file()": "INPUT_PATH = os.getenv('SPREADSHEET_LLM_INPUT_PATH', '../working/data/occupation-standardization/occupation_desc.csv')",
        "GGUF_MODEL_PATH = globals().get('GGUF_MODEL_PATH', '/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-qat-GGUF/snapshots/c1f25db7cf31985b52caa1db777eb72d17ca1c7c/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')": "GGUF_MODEL_PATH = os.getenv('GGUF_MODEL_PATH', globals().get('GGUF_MODEL_PATH', ''))",
    },
}


def sanitize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    replacements = REPLACEMENTS.get(path.name, {})
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        text = source if isinstance(source, str) else "".join(source)
        for old, new in replacements.items():
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        sanitize(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
