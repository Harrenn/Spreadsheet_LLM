# Spreadsheet LLM

Private tools for processing spreadsheet rows with API-hosted or local GGUF language models.

The repository contains two projects:

1. **Branch Cleaner** — fills missing branch coordinates, parses Philippine addresses into structured columns, and preserves the original workbook or CSV data.
2. **Occupation Standardization** — classifies free-text occupation descriptions into a controlled subgroup taxonomy, one row at a time.

## Repository layout

```text
branch_cleaner/   Reusable interactive Python cleaner
notebooks/        Colab, Vast AI, and CUDA notebook workflows
docs/             Tool-specific usage and security notes
examples/         Tiny synthetic input files
tests/            Offline Branch Cleaner and repository checks
working/          Ignored real inputs, outputs, models, and downloads
```

## Local setup

Python 3.11 or 3.12 is recommended. `llama-cpp-python` installation varies by CPU/GPU platform; the hosted notebooks include environment-specific installation cells.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Branch Cleaner

Run it against a directory containing one or more CSV/XLSX inputs:

```bash
python -m branch_cleaner.branch_sheet_cleaner --directory examples/branch_cleaner
python -m branch_cleaner.branch_sheet_cleaner --directory working/data/branch-cleaner
```

The interactive flow lets you choose a file, operation, worksheet columns, and either Gemini or a local GGUF parser. Credentials are requested at runtime and are not stored.

See [docs/branch-cleaner.md](docs/branch-cleaner.md).

## Occupation Standardization

Open `notebooks/occupation_standardization_cuda.ipynb` in a CUDA-capable Jupyter environment. Set `SPREADSHEET_LLM_INPUT_PATH` and `GGUF_MODEL_PATH`, review the taxonomy/configuration cell, test with a small row limit, and then run the full file.

See [docs/occupation-standardization.md](docs/occupation-standardization.md).

## Hosted notebooks

- `branch_cleaner_colab.ipynb` targets Google Colab paths such as `/content/...`.
- `branch_cleaner_vast_ai.ipynb` targets Vast AI paths such as `/workspace/...`.
- API keys and Hugging Face tokens are read from environment variables or entered only in the active session.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite uses fake geocoders/parsers and temporary files; it does not call Google, Gemini, Hugging Face, or live LLM services.
