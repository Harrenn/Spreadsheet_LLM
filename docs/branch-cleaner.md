# Branch Cleaner

## Capabilities

- Read CSV or XLSX input.
- Select the worksheet and source columns interactively.
- Fill rows with missing or invalid latitude/longitude using Google Geocoding.
- Parse full addresses with Gemini or a local GGUF model through `llama-cpp-python`.
- Append fresh `barangay2`, `city_municipality2`, `province2`, `region2`, `zip_code2`, `country2`, and `island_group2` columns.
- In the Colab and Vast AI notebooks, derive `region2` from the PSGC export by exact `province2` matching instead of asking the LLM to classify the region.
- Save a new `_cleaned` file without overwriting the source.

Existing valid coordinate pairs are preserved. Rows without addresses, geocoding misses, and parser failures are summarized rather than silently discarded.

## Run

```bash
python -m branch_cleaner.branch_sheet_cleaner --directory working/data/branch-cleaner
```

You can alternatively set `BRANCH_CLEANER_WORKDIR` and omit `--directory`.

For coordinate cleaning, the Google key is entered through a hidden prompt. For Gemini parsing, the Gemini key is also entered at runtime. Local parsing can select an existing GGUF or search/download one from Hugging Face; model files belong under ignored working storage.

## Input expectations

Headers must be in the first row. Column names are not hardcoded in the local interactive script because the user selects them. Hosted notebooks use configured exact names and should be reviewed before running.

Address parsing in either hosted notebook also requires `PSGC_PATH` (or `SPREADSHEET_LLM_PSGC_PATH`) to point to the exported `philippines_psgc_barangays.csv`. The CSV must contain exact `province_name` and `region_name` headers. Missing or ambiguous PSGC data stops the parse before the model is loaded. Unmatched parsed provinces are left blank in `region2` and reported with counts and sample worksheet rows; no fuzzy match is attempted.

## Outputs

CSV inputs remain CSV; XLSX inputs preserve non-selected worksheets. When a cleaned filename already exists, a numeric suffix is added.
