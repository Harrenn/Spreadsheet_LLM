# Branch Cleaner

## Capabilities

- Read CSV or XLSX input.
- Select the worksheet and source columns interactively.
- Fill rows with missing or invalid latitude/longitude using Google Geocoding.
- Parse full addresses with Gemini or a local GGUF model through `llama-cpp-python`.
- Append fresh `barangay2`, `city_municipality2`, `province2`, `region2`, `zip_code2`, `country2`, and `island_group2` columns.
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

## Outputs

CSV inputs remain CSV; XLSX inputs preserve non-selected worksheets. When a cleaned filename already exists, a numeric suffix is added.
