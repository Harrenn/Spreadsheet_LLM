# Occupation Standardization

The CUDA notebook converts free-text `occupation_desc` values into one controlled `subgroup` value using a local GGUF instruction model.

## Workflow

1. Set `SPREADSHEET_LLM_INPUT_PATH` to a CSV or XLSX file.
2. Set `GGUF_MODEL_PATH` to a compatible local model.
3. Review `INPUT_TEXT_COLUMN`, `OUTPUT_COLUMNS`, taxonomy, context size, GPU layers, and row limit.
4. Run a small `PROCESS_ROW_LIMIT` first.
5. Inspect JSON parsing and classification quality.
6. Set the limit to `None` for a full run.

The notebook processes nonblank rows individually, requires compact JSON, validates the configured output key, appends a fresh output column, and writes a suffixed copy beside the input.

## Taxonomy

The current taxonomy covers management, administration, security, logistics, customer service, maintenance, food service, agriculture, domestic support, engineering, skilled trades, education, field operations, accounting, public safety/government, legal, healthcare, media, creative work, production, sales/marketing, IT support, sports, and an explicit `Others/Unspecified` fallback.

Classification output should be reviewed before downstream operational use. Vague entries are intentionally mapped conservatively rather than enriched with unsupported assumptions.
